"""Bounded TVMScript AST interpreter; never eval/exec/imports input source."""
from __future__ import annotations

import ast
import itertools
from contextlib import nullcontext

import numpy as np

from .execution import (Buffer, Budget, Expr, LoopBreak, SemanticFault, Unsupported, apply, cast, concrete_index,
                        dtype, extern_scalar, scalar_type, select, tir_scalar, trace_value)
from .capabilities import TIR_METADATA as METADATA, TIR_CALLS as CALLS, TIR_SCALARS as SCALARS
from .capabilities import TIR_BATCH_INFIX, TIR_BATCH_SCRIPT, TIR_BATCH2_SCRIPT, TIR_BATCH2_TILES, TIR_INTRINSICS
from .capabilities import TIR_BREAKS, TIR_CUMSUM, TIR_EXTERN, TIR_EXTERN_SIGNATURES, TIR_SERIAL_MARKERS, TIR_TAIL_STATEMENTS, TIR_TUPLE
from .tir_tail_contracts import WARP_FENCE, SWIZZLE_ATTR, constant, marker_error, script_tail_issues, tail_call, tuple_error
from .tile_ops import (GEMM_OPS, accumulate_matrix, check_hints, copy_region, cumsum_region, fill_region, gemm_controls, reduce_region,
                       gemm_flag, gemm_view, matmul_shape, matrix_step, reduction_details, region_descriptor)


def name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return name(node.value) + "." + node.attr
    return ""


class TIRInterpreter:
    def __init__(self, source, *, max_steps=2_000_000, metadata=None):
        self.metadata=metadata
        try:
            self.tree = ast.parse(source)
        except SyntaxError as exc:
            raise Unsupported(f"TIR Python 语法不可解析：{exc}") from exc
        funcs = [n for n in ast.walk(self.tree) if isinstance(n, ast.FunctionDef)
                 and any(name(d) == "T.prim_func" for d in n.decorator_list)]
        if len(funcs) != 1:
            raise Unsupported("执行后端只接受一个 @T.prim_func")
        self.function = funcs[0]
        self.max_steps = max_steps
        issues=script_tail_issues(self.function)
        if issues:
            node,_,message=issues[0]
            raise Unsupported(f'L{node.lineno}: {message}')
        self._preflight(self.function.body)
        for node in self.tree.body:
            if not isinstance(node, (ast.ClassDef, ast.ImportFrom)):
                raise Unsupported(f"L{node.lineno}: 不支持的 TIR 顶层语句")

    def _preflight(self, statements):
        for stmt in statements:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call) and name(stmt.value.func) in METADATA:
                continue
            if isinstance(stmt, ast.With):
                if len(stmt.items) != 1 or name(stmt.items[0].context_expr.func) != "T.block":
                    raise Unsupported(f"L{stmt.lineno}: 不支持的 with")
                self._preflight(stmt.body)
            elif isinstance(stmt, (ast.For, ast.If)):
                self._preflight(stmt.body)
                self._preflight(stmt.orelse)
                expr = stmt.iter if isinstance(stmt, ast.For) else stmt.test
                self._preflight_expr(expr)
            elif isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Expr)):
                for node in ast.walk(stmt):
                    if isinstance(node, ast.Call) and name(node.func) not in CALLS | SCALARS:
                        raise Unsupported(f"L{node.lineno}: 未支持调用 {name(node.func)}")
                self._preflight_expr(stmt)
            elif isinstance(stmt, ast.Pass):
                pass
            else:
                raise Unsupported(f"L{stmt.lineno}: 未支持语句 {type(stmt).__name__}")

    def _preflight_expr(self, expr):
        for node in ast.walk(expr):
            if isinstance(node, ast.Call) and name(node.func) not in CALLS | SCALARS:
                raise Unsupported(f"L{node.lineno}: 未支持调用 {name(node.func)}")
            if (isinstance(node,ast.BinOp) and type(node.op).__name__ not in TIR_BATCH_INFIX
                    and type(node.op) not in {ast.Add,ast.Sub,ast.Mult,ast.Div,ast.FloorDiv,ast.Mod}):
                raise Unsupported(f"L{node.lineno}: 未支持二元运算 {type(node.op).__name__}")
            if (isinstance(node,ast.UnaryOp) and not isinstance(node.op,(ast.USub,ast.Not,ast.Invert))):
                raise Unsupported(f"L{node.lineno}: 未支持的布尔/一元表达式")
            if isinstance(node,ast.Compare) and (len(node.ops)!=1 or type(node.ops[0]) not in {ast.Eq,ast.NotEq,ast.Lt,ast.LtE,ast.Gt,ast.GtE}):
                raise Unsupported(f"L{node.lineno}: 未支持的比较")
            if isinstance(node,ast.Call) and name(node.func)=='T.call_intrin':
                if (len(node.args)<2
                        or not all(isinstance(a,ast.Constant) and isinstance(a.value,str) for a in node.args[:2])
                        or node.args[1].value not in TIR_INTRINSICS
                        or (node.keywords and (node.args[1].value not in TIR_BATCH2_TILES or
                                               any(k.arg!='annotations' for k in node.keywords)))):
                    raise Unsupported(f'L{node.lineno}: T.call_intrin 只支持明确列出的字面量标量操作与 TileOp')
            if isinstance(node,ast.Call) and name(node.func)=='T.match_buffer':
                if any(k.arg not in {'dtype','strides'} for k in node.keywords):
                    raise Unsupported(f"L{node.lineno}: 未支持 match_buffer 附加属性")

    def run(self, buffers, symbols=None, symbolic=False):
        self.budget = buffers[0].budget if buffers else Budget(self.max_steps)
        self.symbolic = symbolic
        self.env = dict(symbols or {})
        self.symbols = dict(symbols or {})
        self.reductions = []
        self.writer_locations = {}
        self.events,self.loop_stack = [],[]
        self.nonunit_thread = False
        self.handles = {}
        self.shared = {}
        self.thread_active = False
        self.used_physical_threads = False
        params = self.function.args.posonlyargs + self.function.args.args
        if len(params) != len(buffers):
            raise Unsupported("TIR 参数数量与测试契约不同")
        self.env.update({p.arg: buf for p, buf in zip(params, buffers)})
        for param,buf in zip(params,buffers):
            annotation=param.annotation
            if isinstance(annotation,ast.Call) and name(annotation.func) in {'T.Buffer','T.Tensor'} and annotation.args:
                buf.tir_static_ones=self.static_ones(annotation.args[0],buf.shape)
        self._bind_dynamic_shapes(buffers)
        self._sequence(self.function.body)
        model = ("logical blocks; one-dimensional cooperative thread simulation"
                 if self.used_physical_threads else "logical_blocks; unused physical thread axes only")
        return {"steps": self.budget.steps, "execution_model": model,
                "reductions": self.reductions, "node_events": self.events}

    def _bind_dynamic_shapes(self, buffers):
        # Only direct dimension symbols can be inferred; expressions require
        # explicit symbols in the case contract, never algebraic guesses.
        for stmt in self.function.body:
            if not isinstance(stmt, ast.Assign) or not isinstance(stmt.value, ast.Call):
                continue
            call = stmt.value
            if name(call.func) != "T.match_buffer" or len(call.args) < 2:
                continue
            buf = self.env.get(name(call.args[0]))
            if not isinstance(buf, Buffer):
                continue
            strides=next((k.value for k in call.keywords if k.arg=='strides'),None)
            if isinstance(strides,(ast.Tuple,ast.List)):
                for expression,size in zip(strides.elts,buf.values.strides):
                    symbol=(expression.id if isinstance(expression,ast.Name) else
                            expression.value if isinstance(expression,ast.Constant) and isinstance(expression.value,str) else None)
                    if symbol:
                        size=int(size//buf.dtype.itemsize)
                        old=self.symbols.setdefault(symbol,size)
                        if old!=size:raise Unsupported('stride 符号契约不一致')
            shape = call.args[1]
            if not isinstance(shape, (ast.Tuple, ast.List)) or len(shape.elts) != len(buf.shape):
                raise Unsupported("match_buffer rank 与参数不同")
            for expression, size in zip(shape.elts, buf.shape):
                if isinstance(expression, ast.Name):
                    old = self.symbols.setdefault(expression.id, size)
                    if old != size:
                        raise SemanticFault(f"动态维度 {expression.id} 不一致")
        self.env.update(self.symbols)

    def _sequence(self, statements):
        for offset, stmt in enumerate(statements):
            self.budget.tick()
            if (isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call)
                    and name(stmt.value.func) == "T.launch_thread"):
                call = stmt.value
                axis, extent = self.eval(call.args[0]), int(self.eval(call.args[1]))
                target = name(stmt.targets[0])
                if not target or extent < 0:
                    raise Unsupported("无效的 launch_thread")
                tail = statements[offset + 1:]
                if axis.startswith("blockIdx."):
                    for value in range(extent):
                        self.env[target] = np.int64(value)
                        with self.budget.at('block', stmt.lineno, variable=target, value=value, extent=extent):
                            self._sequence(tail)
                    return
                if not axis.startswith("threadIdx."):
                    raise Unsupported(f"未知线程轴 {axis}")
                used = self._runtime_names(tail)
                if target in used and extent != 1:
                    if self.thread_active:
                        raise Unsupported("当前只支持一维物理线程组")
                    self.used_physical_threads = True
                    self.physical(tail,target,extent)
                    return
                self.env[target] = np.int64(0)
                if extent!=1:
                    old=self.nonunit_thread
                    self.nonunit_thread=True
                    try: self._sequence(tail)
                    finally: self.nonunit_thread=old
                    return
                continue
            self.statement(stmt)

    def _runtime_names(self, statements):
        names = set()
        for stmt in statements:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call) and name(stmt.value.func) in METADATA:
                continue
            if isinstance(stmt, ast.With):
                names.update(self._runtime_names(stmt.body))
            else:
                names.update(n.id for n in ast.walk(stmt) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load))
        return names

    def statement(self, node):
        with self.budget.at('statement', node.lineno):
            return self._statement(node)

    def _statement(self, node):
        if isinstance(node, ast.With):
            self._sequence(node.body)
        elif isinstance(node, ast.Assign):
            if len(node.targets) != 1:
                raise Unsupported("不支持多目标赋值")
            value = self.eval(node.value)
            self.assign(node.targets[0], value, node.lineno)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            value=self.eval(node.value)
            kind=name(node.annotation)
            if kind.startswith("T."):
                value=cast(value,kind[2:])
            else:
                raise Unsupported("未支持的标量类型注解")
            self.assign(node.target,value,node.lineno)
        elif isinstance(node, ast.AugAssign):
            op = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul"}.get(type(node.op))
            if op is None:
                raise Unsupported("不支持的增量赋值")
            self.assign(node.target, apply(op, [self.eval(node.target), self.eval(node.value)]), node.lineno)
        elif isinstance(node, ast.Expr):
            if isinstance(node.value, ast.Call) and name(node.value.func) in METADATA:
                if name(node.value.func) == "T.block_attr":
                    payload=node.value.args[0]
                    if not isinstance(payload,ast.Dict):raise Unsupported('block_attr 必须为字典')
                    for key,value in zip(payload.keys,payload.values):
                        if isinstance(key,ast.Constant) and key.value=='tl.local_var_init':
                            if not isinstance(value,ast.Dict):raise Unsupported('local_var_init 必须为字典')
                            for handle,initial in zip(value.keys,value.values):self.handles[self.eval(handle)]=self.eval(initial)
                        else:raise Unsupported(f'L{node.lineno}: 未支持 block_attr 属性')
                if name(node.value.func) == "T.attr" and len(node.value.args) > 1 and isinstance(node.value.args[1], ast.Constant) and node.value.args[1].value == "tl.assume":
                    if not bool(self.eval(node.value.args[0])):
                        raise Unsupported(f"L{node.lineno}: 测试不满足 TIR assumption")
                if (name(node.value.func)=='T.attr' and len(node.value.args)==3
                        and constant(node.value.args[1])==SWIZZLE_ATTR and tail_call(node.value.args[2])):
                    self.tail_statement(node.value.args[2],metadata=True)
                return
            value=node.value
            if isinstance(value,ast.Call) and name(value.func)=='T.evaluate':
                if len(value.args)!=1 or value.keywords:
                    raise Unsupported('T.evaluate 需要一个表达式')
                value=value.args[0]
            if not self.tail_statement(value):
                self.eval(value)
        elif isinstance(node, ast.For):
            if node.orelse:
                raise Unsupported("不支持循环 else")
            call = node.iter
            if not isinstance(call, ast.Call) or name(call.func) not in {"range", "T.serial", "T.parallel", "T.unroll", "T.vectorized"}:
                raise Unsupported("不支持的循环迭代器")
            bounds = [int(self.eval(v)) for v in call.args]
            kind=name(call.func)
            region=self.budget.new_region() if kind=='T.parallel' else None
            target=name(node.target)
            if not isinstance(node.target,ast.Name): raise Unsupported('For 循环变量必须是名称')
            missing=object()
            old=self.env.get(target,missing)
            self.loop_stack.append(kind)
            try:
                for value in range(*bounds):
                    self.assign(node.target, np.int64(value), node.lineno)
                    with self.budget.iteration(region,value) if region is not None else nullcontext():
                        with self.budget.at('loop', node.lineno, variable=target,value=value,bounds=bounds):
                            self._sequence(node.body)
            except LoopBreak:
                if kind not in {'T.serial','range'}: raise
            finally:
                self.loop_stack.pop()
                if old is missing: self.env.pop(target,None)
                else: self.env[target]=old
        elif isinstance(node, ast.If):
            cond = self.eval(node.test)
            if isinstance(cond, Expr):
                raise Unsupported("符号执行暂不支持有副作用的输入相关 if；支持标量 if_then_else")
            with self.budget.at('guard', node.lineno, condition=ast.unparse(node.test), value=bool(cond)):
                self._sequence(node.body if bool(cond) else node.orelse)
        elif isinstance(node, ast.Pass):
            return
        else:
            raise Unsupported(f"L{node.lineno}: 不支持语句")

    def assign(self, target, value, line):
        if isinstance(target, ast.Name):
            if value is None:
                if target.id not in self.symbols:
                    raise Unsupported(f"动态符号 {target.id} 未绑定")
                value = self.symbols[target.id]
            self.env[target.id] = value
            if isinstance(value, Buffer):
                value.name = target.id
        elif isinstance(target, ast.Tuple) and isinstance(value, tuple):
            for a, b in zip(target.elts, value):
                self.assign(a, b, line)
        elif isinstance(target, ast.Subscript):
            buf, idx = self.address(target)
            buf.write(idx, value, line)
        else:
            raise Unsupported("不支持的赋值目标")

    def address(self, node):
        buf = self.eval(node.value)
        if not isinstance(buf, Buffer):
            raise Unsupported("非 buffer 下标")
        elements = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        if any(isinstance(n, ast.Slice) for n in elements):
            raise Unsupported("当前 TIR 后端仅支持标量下标与 T.region 矩形拷贝")
        return buf, tuple(self.eval(i) for i in elements)

    def eval(self, node):
        try:
            value = self._eval(node)
        except SemanticFault as exc:
            if exc.line is None:
                exc.line = node.lineno
            raise
        if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.Compare, ast.Call)):
            return trace_value(value, ast.unparse(node)[:160], node.lineno,
                               name(node.func) if isinstance(node, ast.Call) else type(node).__name__)
        return value

    def _eval(self, node):
        self.budget.tick()
        if isinstance(node, ast.Constant):
            if isinstance(node.value, float):
                return np.float32(node.value)
            if type(node.value) is int:
                return np.int64(node.value)
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in self.env:
                raise Unsupported(f"L{node.lineno}: 未绑定名称 {node.id}")
            return self.env[node.id]
        if isinstance(node, (ast.Tuple, ast.List)):
            return tuple(self.eval(n) for n in node.elts)
        if isinstance(node, ast.Subscript):
            buf, idx = self.address(node)
            return buf.read(idx, node.lineno)
        if isinstance(node, ast.BinOp):
            if type(node.op).__name__ in TIR_BATCH_INFIX:
                a,b = self.eval(node.left),self.eval(node.right)
                return tir_scalar(TIR_BATCH_INFIX[type(node.op).__name__], [a,b], scalar_type(a))
            op = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div", ast.FloorDiv: "floordiv", ast.Mod: "mod"}.get(type(node.op))
            if op is None:
                raise Unsupported("不支持的二元运算")
            return apply(op, [self.eval(node.left), self.eval(node.right)])
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Invert):
            value = self.eval(node.operand)
            return tir_scalar('tir.bitwise_not', [value], scalar_type(value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return apply("neg", [self.eval(node.operand)])
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            op = {ast.NotEq: "ne", ast.Eq: "eq", ast.Lt: "lt", ast.LtE: "le", ast.Gt: "gt", ast.GtE: "ge"}.get(type(node.ops[0]))
            if op is None:
                raise Unsupported("不支持的比较")
            return apply(op, [self.eval(node.left), self.eval(node.comparators[0])])
        if isinstance(node, ast.BoolOp):
            values=[]
            for item in node.values:
                value=self.eval(item)
                values.append(value)
                if not isinstance(value,Expr):
                    if isinstance(node.op,ast.And) and not bool(value): return np.bool_(False)
                    if isinstance(node.op,ast.Or) and bool(value): return np.bool_(True)
            from functools import reduce
            return reduce(lambda a,b:apply('and' if isinstance(node.op,ast.And) else 'or',[a,b]),values)
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.Not):
            return apply('not',[self.eval(node.operand)])
        if isinstance(node, ast.Call):
            return self.call(node)
        raise Unsupported(f"L{getattr(node,'lineno','?')}: 不支持表达式 {type(node).__name__}")

    def call(self, node):
        function = name(node.func)
        tail=tail_call(node)
        if tail is not None:
            op,args,target=tail
            if op!=TIR_EXTERN:
                raise Unsupported(f'{op}: 只能作为语句使用')
            symbol=constant(args[0]) if args else None
            signature=TIR_EXTERN_SIGNATURES.get(symbol)
            if node.keywords or signature is None or target!=signature[0] or len(args)!=1+len(signature[1]):
                raise Unsupported(f'tir.call_extern {symbol!r}: 未支持的符号、参数或返回 dtype')
            values=[self.eval(a) for a in args[1:]]
            return extern_scalar(symbol,values,target,[scalar_type(v) for v in values])
        if function in {'T.min','T.max'}:
            values=[self.eval(a) for a in node.args]
            if len(values)!=2:raise Unsupported('min/max 需要两个标量')
            if not self.symbolic and any((np.asarray(v).dtype.kind=='f' or np.asarray(v).dtype.name=='bfloat16') and (np.isnan(v) or (v==0 and np.signbit(v))) for v in values):
                raise Unsupported('min/max NaN/负零规则尚未核实')
            return apply(function[2:],values)
        if function == 'T.call_intrin':
            kind, operation = (arg.value for arg in node.args[:2])  # checked in preflight
            if operation in TIR_BATCH2_TILES or operation == 'tl.tileop.region':
                if kind != 'handle':
                    raise Unsupported('TileOp call_intrin 需要 handle 类型')
                if operation == 'tl.tileop.region':
                    return self.explicit_tile_region(node.args[2:])[0]
                return self.tile_call(operation,node,node.args[2:],raw=True)
            values = [self.eval(a) for a in node.args[2:]]
            if operation in ('tl.ieee_fdiv','tl.ieee_fmaf','tl.ieee_frcp') and (not isinstance(node.args[-1],ast.Constant) or node.args[-1].value!='rn'):
                raise Unsupported(f'L{node.lineno}: {operation} 需要字面量舍入模式 rn')
            return tir_scalar(operation, values, kind)
        if function in TIR_BATCH_SCRIPT:
            operation = TIR_BATCH_SCRIPT[function]
            values = [self.eval(a) for a in node.args]
            if operation == 'tir.reinterpret':
                if len(values)==1 and len(node.keywords)==1 and node.keywords[0].arg=='dtype':
                    target = self.eval(node.keywords[0].value)
                elif len(values)==2 and not node.keywords:
                    value,target = values if isinstance(values[1],str) else tuple(reversed(values))
                    values = [value]
                else:
                    raise Unsupported('T.reinterpret 需要一个值和一个类型字符串')
                if not isinstance(target,str):
                    raise Unsupported('T.reinterpret 目标需要类型字符串')
                return tir_scalar(operation, values, target)
            if node.keywords:
                raise Unsupported(f'{function} 不支持附加属性')
            if operation == 'tl.infinity':
                if len(values)!=1 or not isinstance(values[0],str):
                    raise Unsupported('T.infinity 需要一个显式浮点类型')
                return tir_scalar(operation, values, values[0])
            if not values:
                raise Unsupported(f'{function} 需要标量输入')
            if operation == 'tl.ieee_frcp' and (len(node.args) != 2 or
                    not isinstance(node.args[1],ast.Constant) or node.args[1].value != 'rn'):
                raise Unsupported(f'{function} 需要一个标量和字面量舍入模式 rn')
            count = {'tl.ieee_fdiv': 2, 'tl.ieee_fmaf': 3}.get(operation)
            if count and len(values)==count:
                values.append('rn')
            elif count and (not isinstance(node.args[-1],ast.Constant) or node.args[-1].value!='rn'):
                raise Unsupported(f'{function} 需要字面量舍入模式 rn')
            kind = 'bool' if operation in ('tir.isnan','tir.isfinite') else scalar_type(values[0])
            return tir_scalar(operation, values, kind)
        if function in {'T.exp','T.exp2','T.rsqrt','T.sqrt','T.sigmoid'}:
            value=self.eval(node.args[0])
            kind=value.type if isinstance(value,Expr) else str(np.asarray(value).dtype)
            return apply(function[2:],[value],kind)
        if function == 'T.handle':
            return ('handle',node.lineno)
        if function == 'T.tvm_storage_sync':
            raise Unsupported(f'L{node.lineno}: barrier 必须在物理线程调度器中执行')
        if function == 'T.reduce':
            return self.reduce(node)
        if function in SCALARS:
            return cast(self.eval(node.args[0]), function[2:]) if node.args else None
        if function == "T.Cast":
            return cast(self.eval(node.args[1]), self.eval(node.args[0]))
        if function == "T.if_then_else":
            cond = self.eval(node.args[0])
            if isinstance(cond, Expr):
                return select(cond, self.eval(node.args[1]), self.eval(node.args[2]))
            with self.budget.at('guard', node.lineno, condition=ast.unparse(node.args[0]), value=bool(cond)):
                return self.eval(node.args[1] if bool(cond) else node.args[2])
        if function == "T.match_buffer":
            buf = self.eval(node.args[0])
            shape = tuple(int(i) for i in self.eval(node.args[1]))
            requested = next((self.eval(k.value) for k in node.keywords if k.arg == "dtype"), "float32")
            if len(node.args) > 2:
                requested = self.eval(node.args[2])
            if shape != buf.shape or dtype(requested) != buf.dtype:
                raise Unsupported("TIR match_buffer 与测试参数 shape/dtype 不一致")
            strides = next((self.eval(k.value) for k in node.keywords if k.arg == "strides"), None)
            if strides is not None:
                strides=tuple(self.symbols.get(s,s) if isinstance(s,str) else s for s in strides)
            actual = tuple(int(i // buf.dtype.itemsize) for i in buf.values.strides)
            if strides is not None and tuple(strides) != actual:
                raise Unsupported("目前仅执行匹配的连续布局")
            buf.tir_static_ones=self.static_ones(node.args[1],buf.shape)
            return buf
        if function == "T.alloc_buffer":
            shape = tuple(int(i) for i in self.eval(node.args[0]))
            self.budget.shape(shape)
            requested = self.eval(node.args[1]) if len(node.args) > 1 else "float32"
            scope=next((self.eval(k.value) for k in node.keywords if k.arg=='scope'),'local')
            data=next((self.eval(k.value) for k in node.keywords if k.arg=='data'),None)
            requested=next((self.eval(k.value) for k in node.keywords if k.arg=='dtype'),requested)
            if any(k.arg not in {'scope','data','dtype'} for k in node.keywords):
                raise Unsupported(f'L{node.lineno}: allocation 属性尚未支持')
            if data is not None:
                if data not in self.handles:raise Unsupported('local.var 未声明初始化')
                initial=self.handles[data]
            else:initial=None
            if self.thread_active and scope.startswith('shared') and node.lineno in self.shared:
                return self.shared[node.lineno]
            buf=Buffer(np.empty(shape, dtype=dtype(requested)), initialized=False,
                          symbolic=self.symbolic, budget=self.budget, allocation_line=node.lineno)
            buf.tir_static_ones=self.static_ones(node.args[0],shape)
            if initial is not None:
                for idx in np.ndindex(shape):buf.write(idx,initial,node.lineno)
            if self.thread_active and scope.startswith('shared'):
                self.shared[node.lineno]=buf
            return buf
        if function == "T.region":
            if node.keywords: raise Unsupported('T.region 不接受关键字参数')
            return self.explicit_tile_region(node.args)[0]
        if function in TIR_BATCH2_SCRIPT:
            return self.tile_call(TIR_BATCH2_SCRIPT[function],node,node.args,
                                  raw=function in {'T.gemm','T.gemm_py','T.wgmma_gemm'} and len(node.args)>=10)
        if function == "T.copy":
            if len(node.args)!=2 or any(k.arg!='disable_tma' or not bool(self.eval(k.value)) for k in node.keywords):
                raise Unsupported(f'L{node.lineno}: 未支持 copy 属性')
            source,target=[self.as_region(self.eval(n)) for n in node.args]
            if any(isinstance(i,Expr) for region in (source,target) for i in region[1]):
                raise Unsupported(f'L{node.lineno}: copy 的数据相关符号下标尚未支持；仍可单独执行具体输入')
            sa=[i for i,n in enumerate(source[2]) if n!=1];ta=[i for i,n in enumerate(target[2]) if n!=1]
            if tuple(source[2][i] for i in sa)!=tuple(target[2][i] for i in ta):
                raise Unsupported('copy 仅支持相同非单位维度，不能猜测 reshape/transpose')
            shape=tuple(source[2][i] for i in sa)
            # RISC-V TileLang scalar copy lowering src/op/copy.cc:390:
            # invalid source lanes become zero; invalid destination lanes skip.
            # Ordinary scalar Buffer.read/write still diagnose out-of-bounds.
            values=[]
            for idx in np.ndindex(shape):
                a=list(source[1]);b=list(target[1])
                for axis,i in zip(sa,idx):a[axis]+=i
                for axis,i in zip(ta,idx):b[axis]+=i
                a,b=tuple(a),tuple(b)
                if not all(0<=i<n for i,n in zip(b,target[0].shape)):continue
                value=(source[0].read(a,node.lineno) if all(0<=i<n for i,n in zip(a,source[0].shape)) else cast(0,str(target[0].dtype)))
                values.append((b,value))
            for idx,value in values:target[0].write(idx,value,node.lineno)
            return None
        if function == "T.assume":
            if not bool(self.eval(node.args[0])):
                raise Unsupported("测试不满足 T.assume")
            return None
        raise Unsupported(f"L{node.lineno}: 未支持调用 {function}")

    def as_region(self,value):
        if isinstance(value,Buffer):return value,(0,)*len(value.shape),value.shape
        if isinstance(value,tuple) and len(value)==3 and isinstance(value[0],Buffer):return value
        raise Unsupported('需要 buffer 或矩形 region')

    def tail_statement(self,node,metadata=False):
        tail=tail_call(node)
        if tail is None or tail[0] not in TIR_TAIL_STATEMENTS:
            return False
        op,args,_=tail
        self.budget.tick()
        if op in TIR_BREAKS:
            if args or self.thread_active or not self.loop_stack or self.loop_stack[-1] not in {'T.serial','range'}:
                raise Unsupported(f'{op}: 只能跳出最内层串行 For')
            raise LoopBreak()
        if op in TIR_SERIAL_MARKERS or op==TIR_TUPLE:
            vals=[constant(a) for a in args]
            if self.thread_active and op in TIR_SERIAL_MARKERS:
                raise Unsupported(f'{op}: 物理线程执行中不能忽略串行提示')
            pointer=(op==WARP_FENCE and len(args)==4 and isinstance(args[1],ast.Attribute)
                     and args[1].attr=='data' and isinstance(self.eval(args[1].value),Buffer))
            err=tuple_error(vals,metadata) if op==TIR_TUPLE else marker_error(op,vals,pointer=pointer)
            if err: raise Unsupported(err)
            self.events.append({'stage_text_line':node.lineno,'action':'serial_noop','operation':op})
            if len(self.events)>128: self.events.pop(0)
            return True
        if op==TIR_CUMSUM:
            if self.thread_active or self.nonunit_thread:
                raise Unsupported('cumsum 仅支持非协作、串行区域')
            if len(args)!=4: raise Unsupported('cumsum 需要 4 个参数')
            dim,reverse=map(constant,args[2:])
            if type(dim) is not int or type(reverse) not in (bool,int) or reverse not in (0,1):
                raise Unsupported('cumsum 需要静态整数 dim 与布尔 reverse')

            def write(buf,index,value,line):
                buf.write(index,value,line)
                if buf.position is not None:
                    self.writer_locations[(buf.position,buf.root_index(index))] = {
                        'stage_text_line':line,'operation':op}
            cumsum_region(*(self.tile_region(a)[0] for a in args[:2]),dim,bool(reverse),node.lineno,write,
                          lambda details:self.reductions.append({'line':node.lineno,**details}))
            return True
        return False

    @staticmethod
    def literal_int(node):
        if isinstance(node,ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.USub):
            v = TIRInterpreter.literal_int(node.operand)
            return -v if v is not None else None
        if (isinstance(node,ast.Call) and name(node.func) in SCALARS
                and name(node.func).startswith(('T.int','T.uint')) and len(node.args)==1 and not node.keywords):
            return TIRInterpreter.literal_int(node.args[0])
        return None

    @staticmethod
    def literal_flag(node):
        if isinstance(node,ast.Call) and name(node.func)=='T.bool' and len(node.args)==1 and not node.keywords:
            node=node.args[0]
        return isinstance(node,ast.Constant) and type(node.value) is bool or TIRInterpreter.literal_int(node) in (0,1)

    def static_ones(self,spec,shape):
        if not isinstance(spec,(ast.Tuple,ast.List)) or len(spec.elts)!=len(shape):
            return ()
        return tuple(i for i,(v,n) in enumerate(zip(spec.elts,shape)) if n==1 and self.literal_int(v)==1)

    def tile_region(self,node):
        if isinstance(node,ast.Subscript):
            buf = self.eval(node.value)
            if not isinstance(buf,Buffer): raise Unsupported('TileOp 需要 buffer')
            axes = node.slice.elts if isinstance(node.slice,ast.Tuple) else [node.slice]
            if len(axes)!=len(buf.shape): raise Unsupported('TileOp region rank 不匹配')
            origins,shape,ones = [],[],[]
            for i,(axis,bound) in enumerate(zip(axes,buf.shape)):
                if isinstance(axis,ast.Slice):
                    a = concrete_index(self.eval(axis.lower)) if axis.lower is not None else 0
                    b = concrete_index(self.eval(axis.upper)) if axis.upper is not None else bound
                    if axis.step is not None and self.literal_int(axis.step)!=1:
                        raise Unsupported('TileOp region 切片只支持静态单位步长')
                    n = b-a
                    static_lo=axis.lower is None or self.literal_int(axis.lower) is not None
                    static_hi=(i in getattr(buf,'tir_static_ones',()) if axis.upper is None
                               else self.literal_int(axis.upper) is not None)
                    if n==1 and static_lo and static_hi:
                        ones.append(i)
                elif isinstance(axis,ast.Call) and name(axis.func)=='T.Ramp':
                    if len(axis.args)!=3 or axis.keywords or self.literal_int(axis.args[1])!=1:
                        raise Unsupported('TileOp region 仅支持单位步长 Ramp')
                    a,n = concrete_index(self.eval(axis.args[0])), self.literal_int(axis.args[2])
                    if n is None or n <= 0: raise Unsupported('Ramp lanes 需要静态正整数')
                    if n==1: ones.append(i)
                else:
                    a,n = concrete_index(self.eval(axis)),1
                    ones.append(i)
                origins.append(a);shape.append(n)
            return region_descriptor(buf,origins,shape),tuple(ones)
        if isinstance(node,ast.Call) and name(node.func)=='T.region' and not node.keywords:
            return self.explicit_tile_region(node.args)
        if (isinstance(node,ast.Call) and name(node.func)=='T.call_intrin' and len(node.args)>=2
                and isinstance(node.args[1],ast.Constant) and node.args[1].value=='tl.tileop.region'):
            if node.keywords or not isinstance(node.args[0],ast.Constant) or node.args[0].value!='handle':
                raise Unsupported('非法 tl.tileop.region 签名')
            return self.explicit_tile_region(node.args[2:])
        value=self.eval(node)
        return region_descriptor(*self.as_region(value)),getattr(value,'tir_static_ones',())

    def explicit_tile_region(self,args):
        if len(args)<2 or not isinstance(args[0],ast.Subscript) or self.literal_int(args[1]) not in (1,2,3):
            raise Unsupported('T.region 需要 buffer 起点和静态 access mask')
        (buf,origin,lanes),_ = self.tile_region(args[0])
        axes = args[0].slice.elts if isinstance(args[0].slice,ast.Tuple) else [args[0].slice]
        if len(args)-2 != len(axes):
            raise Unsupported('T.region rank 与显式 extent 不符')
        sizes,ones = [],[]
        for i,(axis,n) in enumerate(zip(axes,args[2:])):
            if isinstance(axis,ast.Slice):
                raise Unsupported('T.region 起点只允许标量下标或 Ramp')
            if isinstance(axis,ast.Call) and name(axis.func)=='T.Ramp':
                explicit=self.literal_int(n)
                if explicit is not None and explicit!=lanes[i]:
                    raise Unsupported('T.region Ramp lanes 与显式 extent 不符')
                sizes.append(lanes[i])
                if lanes[i]==1: ones.append(i)
            else:
                sizes.append(self.eval(n))
                if self.literal_int(n)==1: ones.append(i)
        return region_descriptor(buf,origin,sizes),tuple(ones)

    def tile_call(self,op,node,args,*,raw=False):
        line = node.lineno

        def write(buf,index,value,line):
            if self.thread_active:
                raise Unsupported('Batch 2 TileOp 仅支持串行区域执行')
            buf.write(index,value,line)
            if buf.position is not None:
                self.writer_locations[(buf.position,buf.root_index(index))] = {
                    'stage_text_line':line,'operation':op}

        keywords = {k.arg:k.value for k in node.keywords}
        if None in keywords or len(keywords)!=len(node.keywords):
            raise Unsupported('TileOp 不支持动态或重复关键字')
        hints = {}
        if 'annotations' in keywords:
            obj = keywords.pop('annotations')
            if not isinstance(obj,ast.Dict):
                raise Unsupported('TileOp annotations 必须为字面量字典')
            for key,val in zip(obj.keys,obj.values):
                if not isinstance(key,ast.Constant) or not isinstance(key.value,str) or key.value in hints:
                    raise Unsupported('TileOp annotations 需要唯一的字符串键')
                hints[key.value] = self.eval(val) if key.value in {
                    'coalesced_width','eviction_policy','is_wgmma','is_tcgen05','use_2cta','disable_tma',
                    'no_implicit_async_commit_wait','is_async_copy','is_tma_copy'} else None
        if op == 'tl.tileop.fill' and len(args)==2 and not keywords:
            check_hints(op,hints)
            fill_region(self.tile_region(args[0])[0],self.eval(args[1]),line,write)
            return None
        if op in {'tl.tileop.async_copy','tl.tileop.tma_copy'} and len(args)==2:
            for key,val in keywords.items():
                if key not in hints:
                    hints[key] = self.eval(val) if key in {'coalesced_width','eviction_policy'} else None
            check_hints(op,hints)
            copy_region(*(self.tile_region(a)[0] for a in args),line,write)
            return None
        if op not in GEMM_OPS or len(args)<3:
            raise Unsupported(f'{op}: 非法 TileOp 参数列表')
        if self.thread_active:
            raise Unsupported('Batch 2 GEMM 仅支持串行区域执行')
        check_hints(op,hints)
        regions,ones = zip(*(self.tile_region(x) for x in args[:3]))
        a,b,c = (gemm_view(reg,drops) for reg,drops in zip(regions,ones))
        if raw:
            if len(args) not in (10,19) or keywords:
                raise Unsupported('GEMM 仅支持 10 参数降级形式或 19 参数序列化形式')
            if (not all(self.literal_flag(args[i]) for i in (3,4))
                    or self.literal_int(args[8]) is None
                    or len(args)==19 and any(self.literal_int(args[i]) is None for i in (14,15,16))):
                raise Unsupported('GEMM transpose/policy/k_pack/wg_wait/mbar 需要支持的静态值')
            ta,tb,dims,clear = gemm_controls([self.eval(v) for v in args[3:]],regions)
        else:
            fields = ('transpose_A','transpose_B','policy','clear_accum','k_pack','mbar')
            if len(args)>3+len(fields) or any(f not in fields for f in keywords):
                raise Unsupported('未知 GEMM 关键字或参数数量')
            for key,val in zip(fields,args[3:]):
                if key in keywords: raise Unsupported(f'GEMM 重复参数 {key}')
                keywords[key] = val
            if (any(key in keywords and not self.literal_flag(keywords[key]) for key in fields[:2])
                    or 'policy' in keywords and self.literal_int(keywords['policy']) not in (0,1,2,3)
                    or 'k_pack' in keywords and self.literal_int(keywords['k_pack'])!=1
                    or 'mbar' in keywords and not (isinstance(keywords['mbar'],ast.Constant) and
                                                     keywords['mbar'].value is None)):
                raise Unsupported('GEMM 需要静态 transpose/policy、k_pack=1 和无 mbar')
            ta,tb,clear = (gemm_flag(self.eval(keywords[k]),k) if k in keywords else False
                            for k in ('transpose_A','transpose_B','clear_accum'))
            dims = None
        shape = matmul_shape(a,b,c,ta,tb,dims)
        step = matrix_step(a,b,c)
        self.reductions.append({'line':line,**reduction_details(c,ta,tb,clear)})
        if clear: fill_region(regions[2],0,line,write)
        accumulate_matrix(a,b,c,ta,tb,shape,line,step,write)
        return None

    def physical(self, statements, variable, extent):
        """Cooperative SPMD with private local allocations and block shared data.

        The scheduler never ignores barriers or infers a missing thread layout.
        Thread-dependent annotated logical loops need a decoded Fragment mapping.
        """
        import copy
        if not 1<=extent<=1024:raise Unsupported('物理线程数超出 1..1024')
        shared={};threads=[];layout_cache={}
        for tid in range(extent):
            child=copy.copy(self);child.env=dict(self.env);child.env[variable]=np.int64(tid)
            child.handles=dict(self.handles);child.thread_active=True;child.shared=shared
            child.layout_cache=layout_cache
            child.tid=tid;child.thread_extent=extent;child.thread_region=self.budget.new_region() if tid==0 else threads[0][0].thread_region
            threads.append((child,child.cooperate(statements)))
        epoch=0
        while True:
            waits=[];done=[]
            for child,iterator in threads:
                with self.budget.iteration((child.thread_region,epoch),child.tid):
                    try:waits.append(next(iterator));done.append(False)
                    except StopIteration:waits.append(None);done.append(True)
            if all(done):break
            if any(done) or len(set(waits))!=1:
                raise SemanticFault('物理线程没有到达同一个 barrier',rule_id='SYNC-DIVERGENT-BARRIER',line=next((v for v in waits if v),None))
            epoch+=1
        self.env.update({k:v for k,v in threads[0][0].env.items() if isinstance(v,Buffer) and k in self.env})

    def cooperate(self,statements):
        for node in statements:
            self.budget.tick()
            if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and name(node.value.func)=='T.tvm_storage_sync':
                if len(node.value.args)!=1 or self.eval(node.value.args[0]) not in {'shared','shared.dyn'}:
                    raise Unsupported(f'L{node.lineno}: 未支持的 barrier scope')
                yield node.lineno
            elif isinstance(node,ast.With):yield from self.cooperate(node.body)
            elif isinstance(node,ast.Assign) and isinstance(node.value,ast.Call) and name(node.value.func)=='T.launch_thread':
                axis=self.eval(node.value.args[0]);extent=int(self.eval(node.value.args[1]))
                if extent!=1 or not axis.startswith('threadIdx.'):
                    raise Unsupported(f'L{node.lineno}: 多维物理线程暂未支持')
                self.assign(node.targets[0],np.int64(0),node.lineno)
            elif isinstance(node,ast.For):
                if any(k.arg=='annotations' and 'parallel_loop_layout' in ast.unparse(k.value) for k in node.iter.keywords):
                    yield from self.fragment_loop(node)
                    continue
                if name(node.iter.func)=='T.parallel':raise Unsupported(f'L{node.lineno}: 物理线程轴参与的 logical parallel 缺少布局映射')
                for value in range(*[int(self.eval(v)) for v in node.iter.args]):
                    self.assign(node.target,np.int64(value),node.lineno)
                    yield from self.cooperate(node.body)
            elif isinstance(node,ast.If):
                cond=self.eval(node.test)
                if isinstance(cond,Expr):raise Unsupported('物理线程中的符号条件未支持')
                yield from self.cooperate(node.body if bool(cond) else node.orelse)
            else:self.statement(node)

    def fragment_loop(self,node):
        from .layout import Fragment
        if self.metadata is None:raise Unsupported(f'L{node.lineno}: 缺少完整 Fragment metadata，不能猜测线程布局')
        attr=next(k.value for k in node.iter.keywords if k.arg=='annotations')
        mapping=next((v for k,v in zip(attr.keys,attr.values) if isinstance(k,ast.Constant) and k.value=='parallel_loop_layout'),None)
        if not isinstance(mapping,ast.Subscript) or not isinstance(mapping.slice,ast.Constant):
            raise Unsupported('Fragment 索引必须为常量')
        fragment=Fragment(self.metadata,mapping.slice.value)
        chain=[node];body=node.body
        while len(body)==1 and isinstance(body[0],ast.For) and name(body[0].iter.func)=='T.parallel':
            chain.append(body[0]);body=body[0].body
        if len(chain)!=2:raise Unsupported('当前 Fragment 需二层完整 logical parallel')
        ranges=[range(*[int(self.eval(v)) for v in n.iter.args]) for n in chain]
        if any(r.start!=0 or r.step!=1 or len(r)!=size for r,size in zip(ranges,fragment.sizes)):
            raise Unsupported('Fragment 输入域与 parallel 循环不一致')
        if node.lineno not in self.layout_cache:
            owners=[[] for _ in range(self.thread_extent)]
            for i,j in itertools.product(*ranges):
                owner=fragment.owner(i,j)
                if not 0<=owner<self.thread_extent:raise Unsupported('Fragment 映射超出物理线程组')
                owners[owner].append((i,j))
            self.layout_cache[node.lineno]=owners
        for i,j in self.layout_cache[node.lineno][self.tid]:
            for n,value in zip(chain,[i,j]):self.assign(n.target,np.int64(value),n.lineno)
            yield from self.cooperate(body)

    def reduction_region(self, node):
        """Keep singleton scalar indices as unit axes, as saved T.reduce does."""
        if isinstance(node, ast.Name):
            buf = self.eval(node)
            if not isinstance(buf, Buffer):
                raise Unsupported(f'L{node.lineno}: 归约参数不是 buffer')
            return buf, (0,) * len(buf.shape), buf.shape
        if not isinstance(node, ast.Subscript):
            raise Unsupported(f'L{node.lineno}: 不支持的归约区域')
        buf = self.eval(node.value)
        if not isinstance(buf, Buffer):
            raise Unsupported(f'L{node.lineno}: 归约参数不是 buffer')
        axes = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        if len(axes) != len(buf.shape):
            raise Unsupported(f'L{node.lineno}: 归约区域 rank 不同')
        starts, sizes = [], []
        for a, bound in zip(axes, buf.shape):
            if isinstance(a, ast.Slice):
                start = int(self.eval(a.lower)) if a.lower is not None else 0
                stop = int(self.eval(a.upper)) if a.upper is not None else bound
                step = int(self.eval(a.step)) if a.step is not None else 1
                if step != 1 or stop < start:
                    raise Unsupported(f'L{node.lineno}: 归约切片只支持正向单位步长')
                starts.append(start)
                sizes.append(stop - start)
            else:
                starts.append(int(self.eval(a)))
                sizes.append(1)
        return buf, tuple(starts), tuple(sizes)

    def reduce(self, node):
        if len(node.args) != 5 or node.keywords:
            raise Unsupported(f'L{node.lineno}: 需要五参数 T.reduce')
        source,target = map(self.reduction_region,node.args[:2])
        kind,axis,clear = [self.eval(a) for a in node.args[2:]]
        if not isinstance(axis,(int,np.integer)) or isinstance(axis,(bool,np.bool_)):
            raise Unsupported(f'L{node.lineno}: 归约轴必须为整数')
        reduce_region(source,target,kind,axis,clear,node.lineno,
                      lambda buf,idx,value,line:buf.write(idx,value,line),
                      lambda info:self.reductions.append({'line':node.lineno,**info}))
