"""Restricted simulation of real tvm.tir nodes; no script()/AST round trip.

TVM owns parsing and structural verification. This class owns the explicitly
listed execution semantics. Unknown nodes/calls/layouts stop, never disappear.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from copy import copy

from math import prod

import numpy as np

from .capabilities import (TIR_BATCH2_TILES,TIR_BATCH_SCALARS,TIR_BREAKS,TIR_CUMSUM,TIR_EXTERN,
                           TIR_SERIAL_MARKERS,TIR_TAIL_STATEMENTS,TIR_TUPLE)
from .execution import Buffer, Budget, Expr, LoopBreak, SemanticFault, Unsupported, apply, cast, concrete_index, dtype, extern_scalar, select, tir_scalar
from .tir_riscv_contracts import (ACCESS, ADDRESS, ATOMIC_LOAD, ATOMIC_STORE, INT, PTYPE,
                                  SCALAR, SHUFFLES, STREAM_SCALAR, STREAM_VECTOR,
                                  PointerView, shuffle_source, stream_words)
from .tir_tail_contracts import SWIZZLE_ATTR, WARP_FENCE, extern_error, marker_error, tuple_error
from .tile_ops import (GEMM_OPS, accumulate_matrix, check_hints, copy_region, cumsum_region, fill_region,
                       gemm_controls, gemm_view, matmul_shape, matrix_step, reduction_details,
                       reduce_region, region_descriptor)


class TVMInterpreter:
    local_init_attr = 'tl.local_var_init'
    storage_sync = 'tir.tvm_storage_sync'
    shared_scopes = {'shared', 'shared.dyn'}
    initializer_dtypes = {'bool','bfloat16','float8_e4m3fn'} | {
        kind + str(bits) for kind,widths in [('int',(8,16,32,64)),('uint',(8,16,32,64)),('float',(16,32,64))]
        for bits in widths}
    binary = {"Add": "add", "Sub": "sub", "Mul": "mul", "Div": "div",
              "FloorDiv": "floordiv", "FloorMod": "mod", "Mod": "rem",
              "EQ": "eq", "NE": "ne", "LT": "lt", "LE": "le", "GT": "gt", "GE": "ge",
              "And": "and", "Or": "or", "Min": "min", "Max": "max"}
    calls = {"tir.exp": "exp", "tir.exp2": "exp2", "tir.log": "log", "tir.sqrt": "sqrt",
             "tir.rsqrt": "rsqrt", "tir.tanh": "tanh", "tir.fabs": "abs", "tir.sigmoid": "sigmoid"}

    def __init__(self, function, *, max_steps=2_000_000):
        from tilelang import tvm
        self.tvm, self.fun, self.max_steps = tvm, function, max_steps
        allowed = set(self.binary) | {'IntImm','FloatImm','StringImm','Var','SizeVar',
            'BufferLoad','Cast','Not','Select','Let','Call','Ramp','SeqStmt','LetStmt',
            'For','IfThenElse','BufferStore','Evaluate','AttrStmt','Block','BlockRealize','BufferRegion','Range'}
        calls = set(self.calls) | set(TIR_BATCH_SCALARS) | TIR_BATCH2_TILES | TIR_TAIL_STATEMENTS | {self.storage_sync,TIR_EXTERN,'tir.if_then_else','tir.assume',
            'tl.tileop.region','tl.tileop.copy','tl.tileop.reduce'}
        calls |= set(SHUFFLES) | {ADDRESS, ACCESS, PTYPE, ATOMIC_LOAD, ATOMIC_STORE}
        self.pointer_buffers = {}
        self.layout_owners = {}
        has_p2_call = False
        gaps,tail_nodes,standalone,blocks,accessed=set(),{},set(),[],{}
        def preflight(node):
            nonlocal has_p2_call
            kind = type(node).__name__
            if kind not in allowed:
                gaps.add(f'TIR node {kind} 尚未支持')
            if kind in {'BufferLoad','BufferStore'}:
                accessed[self.key(node.buffer)] = node.buffer
            op = self.op_name(node)
            if (op in SHUFFLES or op in {ADDRESS,ACCESS,PTYPE,ATOMIC_LOAD,ATOMIC_STORE,'tl.ieee_frcp'}
                    or op==TIR_EXTERN and len(node.args)
                    and self.constant(node.args[0]) in STREAM_SCALAR | STREAM_VECTOR):
                has_p2_call = True
            if kind == 'Call' and op not in calls:
                gaps.add(f'TIR Call {op or node.op} 尚未支持（包括未执行分支）')
            if op in TIR_TAIL_STATEMENTS or op==TIR_EXTERN:
                problem=self.tail_error(node,op)
                if problem: gaps.add(problem)
                if op in TIR_TAIL_STATEMENTS: tail_nodes[self.key(node)]=op
            if kind == 'Block':
                blocks.append(node)
                try:
                    self.check_block_metadata(node,'Block')
                except Unsupported as exc:
                    gaps.add(str(exc))
            if kind=='Evaluate' and self.op_name(node.value) in TIR_TAIL_STATEMENTS:
                standalone.add(self.key(node.value))
            if op == self.storage_sync:
                try: self.sync_scope(node)
                except Unsupported as exc: gaps.add(str(exc))
                tail_nodes[self.key(node)] = op
            if kind == 'Evaluate' and self.op_name(node.value) == self.storage_sync:
                standalone.add(self.key(node.value))
            if kind == 'AttrStmt' and self.op_name(node.value) == self.storage_sync:
                gaps.add(f'{self.storage_sync}: 屏障仅允许作为独立 Evaluate')
            if kind=='AttrStmt' and str(node.attr_key)==SWIZZLE_ATTR and self.op_name(node.value)==TIR_TUPLE:
                standalone.add(self.key(node.value))
                problem=self.tail_error(node.value,TIR_TUPLE,metadata=True)
                if problem: gaps.add(problem)
            # Also reject reuse of the same Call object in a value context.
            scalar_fields={
                'BufferLoad':('indices','predicate'),'BufferStore':('value','indices','predicate'),
                'For':('min','extent'),'IfThenElse':('condition',),'BlockRealize':('iter_values','predicate'),
                'Select':('condition','true_value','false_value'),'Let':('value','body'),'LetStmt':('value',),
                'Cast':('value',),'Not':('a',),'Ramp':('base','stride','lanes'),'Call':('args',),
                'Range':('min','extent'),
            }.get(kind,('a','b') if kind in self.binary else ())
            for field in scalar_fields:
                value=getattr(node,field,None)
                vals=value if field in {'args','indices','iter_values'} and value is not None else (value,)
                for v in vals:
                    if self.op_name(v) in TIR_TAIL_STATEMENTS | {self.storage_sync}:
                        gaps.add(f'{self.op_name(v)}: 仅支持 Evaluate 或允许的元数据，不能用作值')
            if op in TIR_BATCH2_TILES:
                if len(node.args) not in ((10,19) if op in GEMM_OPS else (2,)):
                    gaps.add(f'{op}: 不支持的位置参数数量')
                try:
                    self.tile_hints(node,op)
                except Unsupported as exc:
                    gaps.add(str(exc))
        tvm.tir.stmt_functor.post_order_visit(function.body,preflight)
        self.shuffle_sites = {}
        self.stream_launches = set()
        self.p2_replay = False
        if has_p2_call:
            try:
                self.inspect_riscv_calls(function.body)
            except Unsupported as exc:
                gaps.add(str(exc))
        for key, buf in accessed.items():
            owner = self.layout_owners.get(self.key(buf.data),key)
            if owner != key:
                gaps.add('Block: layout_map 不支持同一 data Var 的不同访存描述符')
        self.initializers = {}
        try:
            self.inspect_initializers(blocks)
        except Unsupported as exc:
            gaps.add(str(exc))
        for key,op in tail_nodes.items():
            if key not in standalone: gaps.add(f'{op}: 仅支持 Evaluate 或允许的元数据，不能用作值')
        self.sync_subtrees = set()
        self.cooperative_launches = set()
        self.launch_shuffle_sites = {}
        self.block_launches_with_sync = set()
        self.p2_replay = bool(self.shuffle_sites or self.stream_launches)
        if self.storage_sync in tail_nodes.values() or self.p2_replay:
            try: self.inspect_sync_tree(function.body)
            except Unsupported as exc: gaps.add(str(exc))
        # Throw outside the FFI callback so TVM does not wrap Unsupported as an
        # internal TVMError and erase the supported/unsupported distinction.
        if gaps: raise Unsupported(sorted(gaps)[0])

    def key(self, node):
        # TVM handles identify nodes; printed variable names need not be unique.
        return hash(node)

    @staticmethod
    def op_name(node):
        return getattr(node.op,'name',None) if type(node).__name__=='Call' else None

    @staticmethod
    def constant(node):
        kind=type(node).__name__
        if kind=='StringImm': return str(node.value)
        if kind=='FloatImm': return float(node.value)
        if kind=='IntImm':
            dt=str(node.dtype)
            return bool(node.value) if dt=='bool' else int(node.value) if dt.startswith(('int','uint')) and 'x' not in dt else None
        return None

    @staticmethod
    def statement_children(node):
        kind = type(node).__name__
        if kind == 'SeqStmt':
            return tuple((f'.seq[{i}]', stmt) for i, stmt in enumerate(node.seq))
        fields = {'BlockRealize':('block',), 'Block':('init','body'),
                  'IfThenElse':('then_case','else_case')}.get(kind, ('body',))
        return tuple((f'.{field}', child) for field in fields
                     if (child := getattr(node, field, None)) is not None)

    def inspect_initializers(self, blocks):
        key = self.local_init_attr
        attrs = self.fun.attrs
        func_map = attrs[key] if attrs is not None and key in attrs else None
        if func_map is None and not any(key in b.annotations for b in blocks):
            return

        def collect(mapping, buffers, path):
            if type(mapping).__name__ != 'Map':
                raise Unsupported(f'{path}: {key} 必须是 Var → PrimExpr 的 Map')
            result = {}
            for var, value in mapping.items():
                matches = [b for b in buffers if b.data.same_as(var)] if type(var).__name__ == 'Var' else []
                if len(matches) != 1 or matches[0].scope() not in {'local', 'local.var'}:
                    raise Unsupported(f'{path}: {key} 的目标必须是本作用域唯一的局部 Buffer.data')
                if any((n := self.constant(s)) is not None and n <= 0 for s in matches[0].shape):
                    raise Unsupported(f'{path}: {key} 的缓冲区没有全零下标元素')
                result[self.key(matches[0])] = (matches[0], value)
            return result

        defaults = collect(func_map, [buf for b in blocks for buf in b.alloc_buffers], 'PrimFunc') if func_map is not None else {}
        scoped = {self.key(b): collect(b.annotations[key], b.alloc_buffers, 'Block')
                  for b in blocks if key in b.annotations}
        bound_shapes = {self.key(dim) for buf in self.fun.buffer_map.values() for dim in buf.shape
                        if type(dim).__name__ in {'Var', 'SizeVar'}}

        def walk(node, bound):
            kind = type(node).__name__
            if kind == 'Block':
                overrides = scoped.get(self.key(node), {})
                for b in node.alloc_buffers:
                    k = self.key(b)
                    for source in (defaults, overrides):
                        if k in source:
                            _, value = source[k]
                            self.check_initializer(value, str(b.dtype), bound)
                            self.initializers[k] = value
            var = (node.var if kind == 'LetStmt' else node.loop_var if kind == 'For' else
                   node.node.var if kind == 'AttrStmt' and str(node.attr_key) == 'thread_extent' else None)
            if var is not None:
                bound = bound | {self.key(var)}
            for _, child in self.statement_children(node):
                walk(child, bound)

        walk(self.fun.body, bound_shapes)

    def check_initializer(self, value, target, bound):
        def fail():
            raise Unsupported(f'{self.local_init_attr}: 初始值必须是作用域内、类型兼容的纯标量表达式')

        def compatible(src, dst):
            return (src in self.initializer_dtypes and dst in self.initializer_dtypes and
                    not (src.startswith(('float','bfloat')) and dst.startswith(('int','uint'))))

        def expr(node, names):
            dt = str(getattr(node, 'dtype', ''))
            if dt not in self.initializer_dtypes: fail()
            kind = type(node).__name__
            if kind in {'IntImm','FloatImm'}: return
            if kind in {'Var','SizeVar'} and self.key(node) in names: return
            if kind == 'Cast' and compatible(str(node.value.dtype), dt):
                expr(node.value, names); return
            if kind == 'Not': expr(node.a, names); return
            if kind in self.binary and (kind != 'Div' or dt.startswith('float')):
                expr(node.a, names); expr(node.b, names); return
            if kind == 'Select':
                for child in (node.condition, node.true_value, node.false_value): expr(child, names)
                return
            if kind == 'Let':
                expr(node.value, names); expr(node.body, names | {self.key(node.var)}); return
            fail()

        if not compatible(str(getattr(value,'dtype','')), target): fail()
        expr(value, bound)

    def sync_scope(self, call):
        args = call.args
        scope = self.constant(args[0]) if len(args) == 1 else None
        if scope not in self.shared_scopes or getattr(call, 'annotations', None):
            raise Unsupported(f'{self.storage_sync}: 只支持单个 shared/shared.dyn 字面量作用域，无分组参数')
        return scope

    def is_sync(self, node):
        return type(node).__name__ == 'Evaluate' and self.op_name(node.value) == self.storage_sync

    def thread_tag(self, node):
        if type(node).__name__ == 'AttrStmt' and str(node.attr_key) == 'thread_extent':
            if type(node.node).__name__ != 'IterVar':
                raise Unsupported(f'{self.storage_sync}: thread_extent 必须绑定 IterVar')
            return str(node.node.thread_tag)
        return ''

    def inspect_sync_tree(self, root):
        def mark(node):
            children = [mark(child) for _, child in self.statement_children(node)]
            if (any(children) or self.is_sync(node) or self.key(node) in self.shuffle_sites
                    or self.key(node) in self.stream_launches):
                self.sync_subtrees.add(self.key(node))
                return True
            return False

        mark(root)

        def inspect(node, blocks=()):
            if self.key(node) not in self.sync_subtrees:
                return
            if (type(node).__name__ == 'For' and getattr(node,'thread_binding',None) is not None
                    or type(node).__name__ == 'AttrStmt' and str(node.attr_key) == 'virtual_thread'):
                raise Unsupported(f'{self.storage_sync}: 仅支持 AttrStmt thread_extent 物理线程分相')
            tag = self.thread_tag(node)
            if tag and tag not in {f'{p}.{a}' for p in ('threadIdx','blockIdx') for a in 'xyz'}:
                raise Unsupported(f'{self.storage_sync}: 未支持线程轴 {tag}')
            if tag.startswith('blockIdx.'):
                blocks = (*blocks, self.key(node))
            if tag.startswith('threadIdx.') and self.constant(node.value) != 1:
                extent = self.constant(node.value)
                if tag != 'threadIdx.x' or type(extent) is not int or not 2 <= extent <= 1024:
                    raise Unsupported(f'{self.storage_sync}: 仅分相静态单一 threadIdx.x（2..1024）')
                self.inspect_cooperation(node.body)
                sites = self.subtree_shuffle_sites(node.body)
                self.launch_shuffle_sites[self.key(node)] = sites
                if sites and extent % 32:
                    raise Unsupported('Warp shuffle: threadIdx.x 需要完整的 32-lane warp')
                self.cooperative_launches.add(self.key(node))
                self.block_launches_with_sync.update(blocks)
                return
            for _, child in self.statement_children(node):
                inspect(child, blocks)

        inspect(root)
        covered = set()
        for launch in self.cooperative_launches:
            covered.update(self.launch_shuffle_sites.get(launch, ()))
        if self.shuffle_sites.keys() - covered:
            raise Unsupported('Warp shuffle: 需要静态、完整且单一的 threadIdx.x 域')

    def inspect_cooperation(self, root):
        buffers = {self.key(b) for b in self.fun.buffer_map.values()}
        owners = {self.key(b.data):self.key(b) for b in self.fun.buffer_map.values()}

        def fail(reason):
            raise Unsupported(f'{self.storage_sync}: {reason}')

        def check_barrier_positions(node):
            if self.key(node) not in self.sync_subtrees:
                return
            kind = type(node).__name__
            if kind == 'IfThenElse':
                if self.subtree_shuffle_sites(node):
                    raise Unsupported('Warp shuffle: 条件分支不能保证一致到达')
                fail('多物理线程条件分支内的屏障尚未支持')
            if kind == 'For':
                step = getattr(node,'step',None)
                loop_kind = int(node.kind)
                warp_loop = bool(self.subtree_shuffle_sites(node))
                if (loop_kind not in ({0,3} if warp_loop else {0}) or type(self.constant(node.min)) is not int
                        or type(self.constant(node.extent)) is not int or self.constant(node.extent) < 0
                        or step is not None and self.constant(step) != 1):
                    fail('多物理线程循环内的屏障需要静态一致、单位步长的串行循环')
            for _,child in self.statement_children(node): check_barrier_positions(child)

        def visit(node, uniform=True):
            kind = type(node).__name__
            has_sync = self.key(node) in self.sync_subtrees
            if kind == 'IfThenElse':
                uniform = False
            elif kind == 'For':
                loop_kind = int(node.kind)
                step = getattr(node,'step',None)
                if (loop_kind not in {0,3} and not self.vector_copy_loop(node)
                        and (loop_kind != 1 or self.constant(node.extent) != 1 or len(node.annotations))):
                    fail('物理线程分相内的逻辑并行布局/向量循环尚未建模')
                self.check_for_metadata(node, '物理线程分相')
                if step is not None and self.constant(step) != 1:
                    fail('物理线程分相内的循环需要单位步长')
                uniform = False  # shared allocations below a loop have a different lifetime
            elif kind == 'AttrStmt':
                tag = self.thread_tag(node)
                if tag:
                    if tag not in {'threadIdx.y','threadIdx.z'} or self.constant(node.value) != 1:
                        fail('多维、重复或嵌套的非单位线程启动尚未建模')
                elif str(node.attr_key) != 'tl.assume' and not self.is_launch_hint(node):
                    fail(f'物理线程分相内未支持 AttrStmt {node.attr_key}')
            elif kind == 'BlockRealize':
                if self.constant(node.predicate) is not True or len(node.iter_values):
                    fail('物理线程分相内的块需要常真谓词、无迭代轴')
            elif kind == 'Block':
                if node.init is not None or len(node.iter_vars) or len(node.match_buffers):
                    fail('物理线程分相内未支持块归约 init、迭代轴或 match_buffers')
                for b in node.alloc_buffers:
                    bkey,dkey = self.key(b),self.key(b.data)
                    if dkey in owners:
                        fail('物理线程分相内未建模同一 data Var 的多个分配或 Buffer 描述符')
                    owners[dkey] = bkey
                    scope = b.scope()
                    if scope in self.shared_scopes:
                        if not uniform or any(type(self.constant(s)) is not int for s in b.shape):
                            fail('共享分配需要无分支、无循环且各线程一致的静态生命周期')
                    elif scope not in {'local','local.var'}:
                        fail(f'物理线程分相内未建模的分配作用域 {scope}')
                    buffers.add(bkey)
            elif has_sync and not self.is_sync(node) and self.key(node) not in self.shuffle_sites and kind not in {'SeqStmt','LetStmt'}:
                fail(f'无法确定 {kind} 中的屏障到达顺序')
            for _, child in self.statement_children(node):
                visit(child, uniform)

        check_barrier_positions(root)
        visit(root)
        problems = set()

        def effects(node):
            kind = type(node).__name__
            if kind in {'BufferLoad','BufferStore'} and self.key(node.buffer) not in buffers:
                problems.add('物理线程分相内不支持捕获未证明作用域的外部局部分配或 Buffer 别名')
            if kind == 'Call' and self.op_name(node) in (TIR_BREAKS | TIR_BATCH2_TILES | TIR_SERIAL_MARKERS |
                    {TIR_CUMSUM,'tl.tileop.copy','tl.tileop.reduce'}):
                problems.add(f'物理线程分相内未建模 {self.op_name(node)} 的跨线程执行方式')

        self.tvm.tir.stmt_functor.post_order_visit(root, effects)
        if problems: fail(sorted(problems)[0])

    @classmethod
    def expression_children(cls, node):
        kind = type(node).__name__
        fields = {
            'BufferLoad':('indices','predicate'), 'BufferStore':('indices','predicate','value'),
            'Let':('value','body'), 'LetStmt':('value',), 'Cast':('value',),
            'For':('min','extent','step'), 'AttrStmt':('value',),
            'IfThenElse':('condition',), 'Select':('condition','true_value','false_value'),
            'BlockRealize':('predicate','iter_values'), 'Evaluate':('value',),
            'Not':('a',), 'Ramp':('base','stride','lanes'), 'Call':('args',),
        }.get(kind,('a','b') if kind in cls.binary else ())
        for field in fields:
            expr = getattr(node,field,None)
            if expr is None:
                continue
            if field in {'indices','iter_values','args'}:
                yield from ((i,child) for i,child in enumerate(expr))
            else:
                yield field,expr

    def check_stream(self, node, symbol):
        vector = symbol in STREAM_VECTOR
        operands = node.args[1:]
        if len(operands) != (2 if vector else 1) or any(self.op_name(x)!=ADDRESS for x in operands):
            raise Unsupported(f'{symbol}: 需要直接的 address_of(BufferLoad) 参数')
        if any(len(x.args)!=1 or type(x.args[0]).__name__!='BufferLoad' for x in operands):
            raise Unsupported(f'{symbol}: 地址参数必须是标量 BufferLoad')
        types = [str(x.args[0].buffer.dtype) for x in operands]
        if any(t not in SCALAR for t in types):
            raise Unsupported(f'{symbol}: 仅支持可按字节访问的标量 Buffer')
        if not vector and (types[0] != str(node.dtype) or dtype(types[0]).itemsize not in (2,4)
                           or symbol == 'tileops::load_f32_evict_last' and types[0] != 'float32'):
            raise Unsupported(f'{symbol}: 需要匹配的 16/32 位标量类型')

    def inspect_riscv_calls(self, root):
        def lane_dependent_expr(expr, dependent):
            if type(expr).__name__ in {'BufferLoad','Let'}:
                return True
            if type(expr).__name__ in {'Var','SizeVar'}:
                return self.key(expr) in dependent
            return any(lane_dependent_expr(child,dependent) for _,child in self.expression_children(expr))

        def walk(node, owners, stmt=None, slot=None, guarded=False, concurrent=False,
                 parent=None, launch=None, lane_dependent=frozenset()):
            kind,op = type(node).__name__,self.op_name(node)
            if stmt is None:
                stmt = node
            if kind == 'Block':
                owners = dict(owners)
                for buf in node.alloc_buffers:
                    k = self.key(buf.data)
                    if k in owners:
                        owners[k] = None  # ambiguous; never guess a descriptor
                    else:
                        owners[k] = buf
            tag = self.thread_tag(node)
            if tag and self.constant(node.value)!=1:
                concurrent = True
                if tag == 'threadIdx.x' and launch is None:
                    launch = node
                    lane_dependent = lane_dependent | {self.key(node.node.var)}
            if kind=='For' and int(node.kind) in {1,2,4} and self.constant(node.extent)!=1:
                concurrent = True
            if op == 'tl.ieee_frcp':
                if (len(node.args)!=2 or self.constant(node.args[1])!='rn'
                        or str(node.dtype) not in {'float32','float64'} or node.args[0].dtype!=node.dtype):
                    raise Unsupported('tl.ieee_frcp: 需要同类型 f32/f64 标量及字面量 rn')
            if op == TIR_EXTERN and len(node.args):
                symbol = self.constant(node.args[0])
                if symbol in STREAM_VECTOR | STREAM_SCALAR:
                    self.check_stream(node,symbol)
                    if launch is not None:
                        self.stream_launches.add(self.key(launch))
                    if symbol in STREAM_VECTOR and (type(parent).__name__!='Evaluate' or slot!='value'):
                        raise Unsupported(f'{symbol}: 16 字节写入必须是独立 Evaluate')
            if op in {ATOMIC_LOAD, ATOMIC_STORE}:
                required = 1 if op==ATOMIC_LOAD else 2
                if (concurrent or len(node.args) not in {required,required+1}
                        or self.op_name(node.args[0]) != ACCESS
                        or len(node.args)>required and (type(self.constant(node.args[-1])) is not int
                            or self.constant(node.args[-1]) not in ({0,1,2,5} if op==ATOMIC_LOAD else {0,3,5}))
                        or op==ATOMIC_STORE and (type(parent).__name__!='Evaluate' or slot!='value'
                            or str(node.args[1].dtype) not in SCALAR)
                        or op==ATOMIC_LOAD and str(node.dtype) not in SCALAR):
                    raise Unsupported(f'{op}: 只支持无并发的有界标量访存、已知内存顺序')
            if op == PTYPE and (self.op_name(parent)!=ACCESS or slot!=0):
                raise Unsupported(f'{PTYPE}: 仅可作 tvm_access_ptr 的元素类型')
            if op == ACCESS:
                if self.op_name(parent) not in {ATOMIC_LOAD,ATOMIC_STORE} or slot!=0:
                    raise Unsupported(f'{ACCESS}: 未建模的消费点 {self.op_name(parent) or type(parent).__name__}')
                args = node.args
                if (len(args)!=5 or str(node.dtype)!='handle' or self.op_name(args[0])!=PTYPE
                        or len(args[0].args) or type(args[1]).__name__!='Var'
                        or str(args[2].dtype) not in INT or type(self.constant(args[3])) is not int
                        or self.constant(args[3])!=1 or type(self.constant(args[4])) is not int
                        or self.constant(args[4]) not in (1,2,3)):
                    raise Unsupported(f'{ACCESS}: 需要明确的标量类型、Buffer.data、整数 offset、extent=1、rw_mask 1/2/3')
                buf = owners.get(self.key(args[1]))
                if buf is None or not buf.data.same_as(args[1]) or buf.dtype!=args[0].dtype:
                    raise Unsupported(f'{ACCESS}: 无法解析同类型、唯一且在作用域内的根 Buffer')
                if not self.constant(args[4]) & (1 if self.op_name(parent)==ATOMIC_LOAD else 2):
                    raise Unsupported(f'{ACCESS}: rw_mask 与消费操作不符')
                self.check_pointer_buffer(buf)
                self.pointer_buffers[self.key(node)] = buf
            if op == ADDRESS:
                sym = self.constant(parent.args[0]) if self.op_name(parent)==TIR_EXTERN and len(parent.args) else None
                if (sym not in STREAM_VECTOR | STREAM_SCALAR
                        or slot not in ({1,2} if sym in STREAM_VECTOR else {1})):
                    raise Unsupported(f'{ADDRESS}: 未建模的消费点 {self.op_name(parent) or type(parent).__name__}')
                if (len(node.args)!=1 or str(node.dtype)!='handle'
                        or type(load:=node.args[0]).__name__!='BufferLoad'
                        or getattr(load,'predicate',None) is not None
                        or any(str(i.dtype) not in INT for i in load.indices)):
                    raise Unsupported(f'{ADDRESS}: 需要无谓词的标量 BufferLoad')
                buf = owners.get(self.key(load.buffer.data))
                if buf is None or not buf.same_as(load.buffer):
                    raise Unsupported(f'{ADDRESS}: Buffer 别名或作用域无法确定')
                self.check_pointer_buffer(buf)
            if op in SHUFFLES:
                if guarded or type(stmt).__name__ not in {'BufferStore','LetStmt'}:
                    raise Unsupported(f'Warp shuffle {op}: 仅支持一致到达的存储 RHS 或 LetStmt')
                if self.key(stmt) in self.shuffle_sites:
                    raise Unsupported('Warp shuffle: 同一语句中只支持一个交换点')
                args = node.args
                if (len(args) not in (4,5) or str(node.dtype) not in SCALAR | {'bool'}
                        or str(args[1].dtype) != str(node.dtype)
                        or str(args[0].dtype) not in {'int32','uint32','int64','uint64'}
                        or str(args[2].dtype) not in INT or type(self.constant(args[3])) is not int
                        or not 1 <= self.constant(args[3]) <= 32
                        or len(args)==5 and self.constant(args[4])!=32):
                    raise Unsupported(f'Warp shuffle {op}: 非法标量类型、mask、静态 width 或 warp_size')
                if SHUFFLES[op]!='idx' and lane_dependent_expr(args[2],lane_dependent):
                    raise Unsupported('Warp shuffle: xor/up/down 的 delta 不能依赖物理 lane 或访存')
                width = self.constant(args[3])
                # The lowerer shifts an i32 mask by candidate_lane even when
                # its predicate is false. Reject widths with incomplete subgroups.
                if 32 % width:
                    raise Unsupported('Warp shuffle: width 必须整除 32，避免掩码移位越界')
                # The source lowering may read a snapshot at the candidate lane
                # using current-lane indices. Admit only unambiguous direct loads
                # with static indices (or literal operands) in the initial model.
                payload = args[1]
                if not (type(payload).__name__ in {'IntImm','FloatImm'} or
                        type(payload).__name__=='BufferLoad'
                        and getattr(payload,'predicate',None) is None
                        and all(type(self.constant(i)) is int for i in payload.indices)):
                    raise Unsupported('Warp shuffle: 仅重放标量常量或固定坐标的直接 BufferLoad')
                self.shuffle_sites[self.key(stmt)] = node,width

            for field,child in self.expression_children(node):
                # Values hoisted for replay may not be taken from lazy
                # branches, expression-local bindings or statement indices.
                hidden = (guarded or kind in {'Let','Select','IfThenElse','For','AttrStmt','BlockRealize','BufferLoad'}
                          or kind=='BufferStore' and field!='value'
                          or op in SHUFFLES or op=='tir.if_then_else')
                walk(child,owners,stmt,field,hidden,concurrent,node,launch,lane_dependent)
            if kind=='LetStmt' and launch is not None and lane_dependent_expr(node.value,lane_dependent):
                lane_dependent = lane_dependent | {self.key(node.var)}
            for _,child in self.statement_children(node):
                # Let expression bodies are handled above as expressions,
                # not as independent statements.
                if stmt is node:
                    walk(child,owners,concurrent=concurrent,launch=launch,lane_dependent=lane_dependent)

        walk(root,{self.key(b.data):b for b in self.fun.buffer_map.values()})

    def check_pointer_buffer(self, buf):
        if (str(buf.dtype) not in SCALAR or not len(buf.shape) or len(buf.axis_separators)
                or self.constant(buf.elem_offset)!=0 or self.key(buf.data) in self.layout_owners):
            raise Unsupported('有界引用需要确定的紧凑标量 Buffer；不支持物理布局重映射或 elem_offset')
        if len(buf.strides):
            shape = tuple(map(self.constant,buf.shape))
            if (any(type(n) is not int or n<0 for n in shape)
                    or tuple(map(self.constant,buf.strides))!=tuple(prod(shape[i+1:]) for i in range(len(shape)))):
                raise Unsupported('有界引用不支持动态或非紧凑 stride')

    def pointer(self, node, path, *, extent=None, mask=None):
        op = self.op_name(node)
        self.budget.tick()
        if op == ADDRESS and extent is not None:
            load = node.args[0]
            buf = self.buffers.get(self.key(load.buffer))
            if buf is None:
                raise Unsupported(f'{path}: address_of Buffer 不在当前作用域')
            indices = tuple(concrete_index(self.expr(i,path+'.index')) for i in load.indices)
            indices = buf.index(indices,0,'write' if mask==2 else 'read')
            offset = sum(a*b for a,b in zip(indices,buf.element_strides))
        elif op == ACCESS and not self.nonunit_threads and not self.budget.parallel and self.key(node) in self.pointer_buffers:
            spec = self.pointer_buffers[self.key(node)]
            buf = self.buffers.get(self.key(spec))
            if buf is None:
                raise Unsupported(f'{path}: access_ptr Buffer 不在当前作用域')
            offset = concrete_index(self.expr(node.args[2],path+'.offset'))
            extent,mask = map(self.constant,node.args[3:5])
        else:
            raise Unsupported(f'{path}: 有界引用仅用于已建模的消费点')
        return PointerView(buf,offset,extent,mask)

    def subtree_shuffle_sites(self, node):
        return ({self.key(node)} if self.key(node) in self.shuffle_sites else set()).union(
            *(self.subtree_shuffle_sites(child) for _,child in self.statement_children(node)))

    def tail_error(self,node,op,metadata=False):
        if getattr(node,'annotations',None):
            return f'{op}: 不支持附加注解'
        args=node.args
        vals=[self.constant(a) for a in args]
        if op==TIR_EXTERN:
            if vals and vals[0] in STREAM_SCALAR | STREAM_VECTOR:
                return None  # Full typed/context checks in inspect_riscv_calls.
            return extern_error(vals,str(node.dtype),[str(getattr(a,'dtype','')) for a in args[1:]])
        if op in TIR_BREAKS and args:
            return f'{op}: 不接受参数'
        if op==TIR_CUMSUM and (len(args)!=4 or type(vals[2]) is not int or
                               type(vals[3]) not in (bool,int) or vals[3] not in (0,1)):
            return 'cumsum 需要 4 参数和静态整数 dim / 布尔 reverse'
        if op==TIR_TUPLE:
            return tuple_error(vals,metadata)
        if op in TIR_SERIAL_MARKERS:
            pointer=(op==WARP_FENCE and len(args)==4 and type(args[1]).__name__=='Var'
                     and str(args[1].dtype)=='handle')
            return marker_error(op,vals,pointer=pointer)
        return None

    def location(self, node, path):
        span = getattr(node, "span", None)
        value = {"node_path": path, "node_type": type(node).__name__, "stage_text_line": None}
        if span is not None:
            value["original_span"] = {"source": str(span.source_name.name), "line": int(span.line),
                                      "column": int(span.column), "end_line": int(span.end_line)}
        return value

    def event(self, node, path, **detail):
        self.events.append({**self.location(node, path), **detail})
        if len(self.events) > 128:
            self.events.pop(0)

    def expr(self, node, path):
        self.budget.tick()
        kind = type(node).__name__
        if kind in {"IntImm", "FloatImm"}:
            return cast(node.value, str(node.dtype))
        if kind == "StringImm":
            return str(node.value)
        if kind in {"Var", "SizeVar"}:
            if self.key(node) not in self.env:
                raise Unsupported(f"{path}: 未绑定变量 {node.name}")
            return self.env[self.key(node)]
        if kind == "Cast":
            return cast(self.expr(node.value, path+".value"), str(node.dtype))
        if kind in self.binary:
            if kind == 'Div' and not str(node.dtype).startswith('float'):
                raise Unsupported(f'{path}: 整数截断除法尚未支持；不经宿主浮点近似')
            values = [self.expr(node.a,path+'.a'),self.expr(node.b,path+'.b')]
            if kind in {'Min','Max'} and str(node.dtype).startswith('float'):
                if any(isinstance(v,Expr) or np.isnan(v) or (v==0 and np.signbit(v)) for v in values):
                    raise Unsupported(f'{path}: 浮点 Min/Max 的符号/NaN/负零规则尚未支持')
            return apply(self.binary[kind], values, str(node.dtype))
        if kind == "Not":
            return apply("not", [self.expr(node.a,path+".a")], "bool")
        if kind == "BufferLoad":
            if getattr(node, "predicate", None) is not None:
                raise Unsupported(f"{path}: predicated BufferLoad 未支持")
            if any(type(i).__name__ == "Ramp" for i in node.indices):
                raise Unsupported(f"{path}: vector load 仅在明确的 tile reduce 区域中支持")
            buf = self.buffers[self.key(node.buffer)]
            index = tuple(self.expr(i, path+f".indices[{j}]") for j,i in enumerate(node.indices))
            self.event(node,path,action="read",buffer=buf.name,index=[str(i) if isinstance(i,Expr) else int(i) for i in index])
            return buf.read(index,0)
        if kind == "Select":
            # Select evaluates both operands; tir.if_then_else below is lazy.
            return select(self.expr(node.condition,path+".condition"), self.expr(node.true_value,path+".true"), self.expr(node.false_value,path+".false"))
        if kind == "Let":
            key = self.key(node.var)
            old = self.env.get(key)
            self.env[key] = self.expr(node.value,path+".value")
            try:
                return self.expr(node.body,path+".body")
            finally:
                if old is None: self.env.pop(key,None)
                else: self.env[key] = old
        if kind == "Call":
            op = getattr(node.op,"name",None)
            if op in SHUFFLES:
                if self.key(node) not in self.shuffle_values:
                    raise Unsupported(f'{path}: Warp shuffle 只能在一致到达的重放点执行')
                return self.shuffle_values[self.key(node)]
            if op == ATOMIC_LOAD:
                return cast(self.pointer(node.args[0], path).read(0), str(node.dtype))
            if op==TIR_EXTERN:
                symbol = self.constant(node.args[0]) if len(node.args) else None
                if symbol in STREAM_SCALAR:
                    self.check_stream(node, symbol)
                    return self.pointer(node.args[1],path,extent=1,mask=1).read(0)
                problem=self.tail_error(node,op)
                if problem: raise Unsupported(problem)
                values=[self.expr(a,path+f'.args[{i}]') for i,a in enumerate(node.args[1:],1)]
                return extern_scalar(self.constant(node.args[0]),values,str(node.dtype),
                                     [str(a.dtype) for a in node.args[1:]])
            if op == "tir.if_then_else" and len(node.args) == 3:
                condition = self.expr(node.args[0],path+".condition")
                if isinstance(condition,Expr):
                    return select(condition,self.expr(node.args[1],path+".true"),self.expr(node.args[2],path+".false"))
                branch = 1 if bool(condition) else 2
                return self.expr(node.args[branch],path+f".args[{branch}]")
            if op in TIR_BATCH_SCALARS:
                values = [self.expr(a, path+f".args[{i}]") for i,a in enumerate(node.args)]
                types = [str(a.dtype) for a in node.args]
                try:
                    return tir_scalar(op, values, str(node.dtype), types)
                except SemanticFault as exc:
                    exc.details.setdefault("tvm_location", self.location(node, path))
                    raise
            if op in self.calls:
                return apply(self.calls[op],[self.expr(a,path+f".args[{i}]") for i,a in enumerate(node.args)],str(node.dtype))
            raise Unsupported(f"{path}: 未支持 Call {op or node.op}")
        raise Unsupported(f"{path}: 未支持表达式 {kind}")

    def region(self, node, path, *, explicit=False):
        if type(node).__name__ != "BufferLoad":
            raise Unsupported(f"{path}: tile 区域必须是 BufferLoad/Ramp")
        if getattr(node,"predicate",None) is not None:
            raise Unsupported(f"{path}: predicated region 未支持")
        origin,shape = [],[]
        for i,index in enumerate(node.indices):
            if type(index).__name__ == "Ramp":
                if type(index.stride).__name__ != 'IntImm' or int(index.stride.value) != 1:
                    raise Unsupported(f"{path}: 仅支持单位步长归约区域")
                if type(index.lanes).__name__ != 'IntImm':
                    raise Unsupported(f'{path}: TIR region 需要静态 Ramp lanes')
                origin.append(concrete_index(self.expr(index.base,path)))
                shape.append(int(index.lanes.value))
            else:
                origin.append(concrete_index(self.expr(index,path+f".indices[{i}]")));shape.append(1)
        return self.buffers[self.key(node.buffer)],tuple(origin),tuple(shape)

    def tile_region(self, node, path):
        kind = type(node).__name__
        if kind == 'BufferLoad':
            region = region_descriptor(*self.region(node,path))
            return region, tuple(i for i,n in enumerate(region[2]) if n==1)
        if kind == 'BufferRegion':
            buf = self.buffers[self.key(node.buffer)]
            origin = tuple(self.expr(r.min,path) for r in node.region)
            extents = [r.extent for r in node.region]
        elif kind == 'Call' and getattr(node.op,'name',None) == 'tl.tileop.region' and len(node.args)>=2:
            load = node.args[0]
            buf,origin,lane_sizes = self.region(load,path+'.args[0]')
            if type(node.args[1]).__name__ != 'IntImm' or int(node.args[1].value) not in range(1,4):
                raise Unsupported(f'{path}: 不支持的 region access mask')
            extents = tuple(i.lanes if type(i).__name__=='Ramp' else provided
                            for i,provided in zip(load.indices,node.args[2:]))
            if len(load.indices)!=len(node.args)-2 or any(
                    type(i).__name__=='Ramp' and type(n).__name__=='IntImm'
                    and int(n.value)!=lanes for i,n,lanes in zip(load.indices,node.args[2:],lane_sizes)):
                raise Unsupported(f'{path}: region rank 或 Ramp lanes 与显式 extent 不符')
        else:
            raise Unsupported(f'{path}: 需要 tl.tileop.region、BufferRegion 或 BufferLoad/Ramp')
        shape = tuple(self.expr(r,path) for r in extents)
        ones = tuple(i for i,n in enumerate(extents) if type(n).__name__=='IntImm' and int(n.value)==1)
        return region_descriptor(buf,origin,shape),ones

    def tile_hints(self,node,op):
        hints = {}
        annotations = getattr(node,'annotations',None)
        if annotations is not None:
            for key,value in annotations.items():
                hints[str(key)] = value.value if type(value).__name__ in {'IntImm','StringImm'} else value
        check_hints(op,hints)

    def tile(self, node, path):
        op = getattr(node.op,"name",None)
        args = node.args
        def write(buf, index, value, line):
            buf.write(index,value,line)
            if buf.position is not None:
                self.writer_locations[(buf.position,buf.root_index(index))] = self.location(node,path)

        if op in TIR_TAIL_STATEMENTS:
            problem=self.tail_error(node,op)
            if problem: raise Unsupported(problem)
            if op in TIR_BREAKS:
                if not self.loop_stack or self.loop_stack[-1]!=0:
                    raise Unsupported(f'{op}: 只能跳出最内层串行 For')
                raise LoopBreak()
            if op in TIR_SERIAL_MARKERS or op==TIR_TUPLE:
                self.event(node,path,action='serial_noop',operation=op)
                return
            if op==TIR_CUMSUM:
                if self.nonunit_threads: raise Unsupported('cumsum 仅支持非协作、串行区域')
                dim,reverse=map(self.constant,args[2:])
                cumsum_region(*(self.tile_region(a,path+f'.args[{i}]')[0] for i,a in enumerate(args[:2])),
                              dim,bool(reverse),0,write,
                              lambda details:self.reductions.append({**self.location(node,path),**details}))
                self.event(node,path,action='tile_cumsum',operation=op)
                return
        if op == ATOMIC_STORE:
            target = self.pointer(args[0],path)
            write(target.buffer,target.index(0,'write'),cast(self.expr(args[1],path),target.buffer.dtype.name),0)
            return
        if op==TIR_EXTERN:
            symbol = self.constant(args[0]) if len(args) else None
            if symbol in STREAM_VECTOR:
                self.check_stream(node,symbol)
                dst_bits,src_bits = (dtype(str(a.args[0].dtype)).itemsize for a in args[1:])
                target = self.pointer(args[1],path,extent=16//dst_bits,mask=2)
                source = self.pointer(args[2],path,extent=16//src_bits,mask=1)
                for i,value in enumerate(stream_words(source,target)):
                    write(target.buffer,target.index(i,'write'),value,0)
                self.event(node,path,action='stream_copy_16_bytes',operation=symbol)
                return
            self.expr(node,path)
            return

        if op in {'tl.tileop.copy','tl.tileop.async_copy','tl.tileop.tma_copy'}:
            if len(args)!=2:
                raise Unsupported(f'{op} 只接受两个位置参数；调度提示必须是已知注解')
            if op != 'tl.tileop.copy': self.tile_hints(node,op)
            source,target = [self.tile_region(a,path+f'.args[{i}]')[0] for i,a in enumerate(args)]
            copy_region(source,target,0,write)
            self.event(node,path,action='tile_copy',operation=op,source=source[0].name,destination=target[0].name)
            return
        if op == 'tl.tileop.fill' and len(args)==2:
            self.tile_hints(node,op)
            dst,_ = self.tile_region(args[0],path+'.args[0]')
            fill_region(dst,self.expr(args[1],path+'.args[1]'),0,write)
            self.event(node,path,action='tile_fill',destination=dst[0].name)
            return
        if op in GEMM_OPS and len(args) in (10,19):
            self.tile_hints(node,op)
            if any(type(args[i]).__name__ != 'IntImm' for i in (3,4,8)):
                raise Unsupported(f'{path}: GEMM transpose 与 warp policy 必须静态')
            if len(args)==19 and (type(args[14]).__name__!='IntImm' or
                                  type(args[15]).__name__!='IntImm' or
                                  type(args[16]).__name__!='IntImm'):
                raise Unsupported(f'{path}: GEMM k_pack/wg_wait 必须静态且不支持 mbar')
            regions,ones = zip(*(self.tile_region(arg,path+f'.args[{i}]') for i,arg in enumerate(args[:3])))
            a,b,c = (gemm_view(reg,drops) for reg,drops in zip(regions,ones))
            ta,tb,dims,clear = gemm_controls([self.expr(arg,path+f'.args[{i}]')
                                            for i,arg in enumerate(args[3:],3)],regions)
            shape = matmul_shape(a,b,c,ta,tb,dims)
            step = matrix_step(a,b,c)
            self.reductions.append({**self.location(node,path),**reduction_details(c,ta,tb,clear)})
            if clear: fill_region(regions[2],0,0,write)
            accumulate_matrix(a,b,c,ta,tb,shape,0,step,write)
            self.event(node,path,action='tile_gemm',operation=op,source=(a.name,b.name),destination=c.name)
            return
        if op == "tl.tileop.reduce" and len(args) == 5:
            source,target = (self.region(a,path+f'.args[{i}]') for i,a in enumerate(args[:2]))
            kind,axis,clear = map(self.constant,args[2:])
            if type(kind) is not str or type(axis) is not int or type(clear) not in (bool,int):
                raise Unsupported(f'{path}: reduce 的 kind / dim / clear 必须是字面量')
            reduce_region(source,target,kind,axis,clear,0,write,
                          lambda info:self.reductions.append({**self.location(node,path),**info}))
            self.event(node,path,action='tile_reduce',source=source[0].name,destination=target[0].name)
            return
        if op == "tir.assume" and len(args) == 1:
            if not bool(self.expr(args[0],path)): raise Unsupported(f"{path}: 输入不满足 assume")
            return
        raise Unsupported(f"{path}: 未支持 Evaluate Call {op}")

    def check_block_metadata(self, block, path):
        if len(block.match_buffers):
            raise Unsupported(f'{path}: Block match_buffers 尚未支持')
        if not len(block.annotations):
            return
        from tilelang.layout import Layout

        for key, value in block.annotations.items():
            if str(key) == self.local_init_attr:
                continue  # Checked with the allocation and lexical bindings in inspect_initializers.
            if str(key) != 'layout_map':
                raise Unsupported(f'{path}: 未支持 Block 注解 {key}')
            if (block.init is not None or len(block.iter_vars) or
                    type(value).__name__ != 'Map'):
                raise Unsupported(f'{path}: layout_map 仅用于无 init、无迭代变量的局部块')
            buffers = {self.key(buf.data):buf for buf in block.alloc_buffers}
            for var,layout in value.items():
                buf = buffers.get(self.key(var))
                if (buf is None or not buf.data.same_as(var) or buf.scope() not in self.shared_scopes
                        or not len(buf.shape) or len(buf.strides) or len(buf.axis_separators)
                        or self.constant(buf.elem_offset) != 0 or type(layout) is not Layout):
                    raise Unsupported(f'{path}: layout_map 需要块内连续 shared/shared.dyn 分配和普通 Layout（非 Fragment）')
                shape = tuple(map(self.constant,buf.shape))
                declared = tuple(map(self.constant,layout.get_input_shape()))
                if any(type(n) is not int or n <= 0 for n in shape) or declared != shape:
                    raise Unsupported(f'{path}: layout_map 输入尺寸须与局部分配一致')
                self.check_layout_mapping(layout,shape,path)
                dkey,bkey = self.key(buf.data),self.key(buf)
                if self.layout_owners.setdefault(dkey,bkey) != bkey:
                    raise Unsupported(f'{path}: layout_map 不支持同一 data Var 的不同分配描述符')

    def check_layout_mapping(self, layout, shape, path):
        # Layout metadata can be ignored for logical accesses only after proving
        # that distinct logical elements cannot share the same physical cell.
        coords = list(layout.get_forward_vars())
        forward = list(layout.get_forward_index())
        if len(coords) != len(shape) or not forward:
            raise Unsupported(f'{path}: layout_map 需要合法的输入输出维度')

        # FFI reconstruction can return different handles for synthetic input
        # placeholders. Their names are scoped to this one Layout descriptor.
        def identifier(var):
            return (str(var.name),str(var.dtype)) if type(var).__name__ == 'Var' else None

        ids = list(map(identifier,coords))
        if None in ids or len(set(ids)) != len(ids):
            raise Unsupported(f'{path}: layout_map 输入占位符不唯一')
        if len(forward) == len(shape) and all(
                identifier(expr) == ids[i] or shape[i] == 1 and self.constant(expr) == 0
                for i,expr in enumerate(forward)):
            # An identity tuple has no arithmetic that can overflow; each
            # individual coordinate still needs a representable signed index.
            if all(dt in ('int32','int64') and n <= 1 << (int(dt[3:])-1)
                   for (_,dt),n in zip(ids,shape)):
                return
            raise Unsupported(f'{path}: layout_map 物理索引类型非法或溢出')
        if len(forward) != 1:
            raise Unsupported(f'{path}: layout_map 仅支持恒等坐标或单索引行主序 packing')

        root = forward[0]
        dt = str(root.dtype)
        coeff = {}

        def linear(expr):
            if str(expr.dtype) != dt:
                raise Unsupported(f'{path}: layout_map 不支持混合位宽下标计算')
            kind = type(expr).__name__
            if kind == 'Add':
                linear(expr.a); linear(expr.b)
                return
            if kind == 'IntImm' and self.constant(expr) == 0:
                return
            if kind == 'Mul' and str(expr.a.dtype) == dt and str(expr.b.dtype) == dt:
                var,weight = identifier(expr.a),self.constant(expr.b)
                if var is None: var,weight = identifier(expr.b),self.constant(expr.a)
            else:
                var,weight = identifier(expr),1
            if var not in ids or var in coeff or type(weight) is not int or weight <= 0:
                raise Unsupported(f'{path}: layout_map 仅支持正步长的行主序线性式；拒绝 swizzle 或重叠')
            coeff[var] = weight

        linear(root)
        footprint = 1
        for dim,var in reversed(list(zip(shape,ids))):
            if dim == 1: continue  # the omitted coordinate is always zero
            stride = coeff.get(var,0)
            if stride < footprint or (footprint == 1 and stride != 1):
                raise Unsupported(f'{path}: layout_map 行主序步长无效或产生重叠')
            footprint += (dim-1)*stride
        # OutputShape() is derived from Forward, not a separately declared
        # physical domain. After a JSON reload it can contain fresh symbolic
        # placeholders. Derive its bound exactly from the static input domain.
        if dt not in ('int32','int64') or footprint > 1 << (int(dt[3:])-1):
            raise Unsupported(f'{path}: layout_map 物理索引类型非法或溢出')

    def is_launch_hint(self, node):
        # CUDA occupancy hint. The RISC-V AttrStmt visitor only visits its body.
        return (self.p2_replay and str(node.attr_key)=='tl.min_blocks_per_sm'
                and type(self.constant(node.value)) is int and self.constant(node.value)>0)

    def vector_copy_loop(self, node):
        if (not self.p2_replay or int(node.kind)!=2 or len(node.annotations)
                or getattr(node,'thread_binding',None) is not None
                or (getattr(node,'step',None) is not None and self.constant(node.step)!=1)
                or type(node.body).__name__!='BufferStore'
                or type(node.body.value).__name__!='BufferLoad'):
            return False
        store,load = node.body,node.body.value
        return (getattr(store,'predicate',None) is None and getattr(load,'predicate',None) is None
                and str(store.buffer.dtype) in SCALAR and store.buffer.dtype==load.buffer.dtype
                and all(str(i.dtype) in INT for i in (*store.indices,*load.indices)))

    def check_for_metadata(self, node, path):
        if not len(node.annotations):
            return
        # RISC-V lowers unrolled loops in original iteration order; this
        # recognized TVMScript flag only controls explicit code expansion.
        if (int(node.kind)==3 and set(map(str,node.annotations.keys())) == {'pragma_unroll_explicit'}
                and self.p2_replay
                and (type(node.annotations['pragma_unroll_explicit']) is bool
                     or type(self.constant(node.annotations['pragma_unroll_explicit'])) is bool)
                and (getattr(node,'step',None) is None or self.constant(node.step)==1)):
            return
        # One pipeline stage preserves the serial loop order in LowerLoopBody.
        # No dependence on pipeline prologues or overlapping stages is modeled.
        if (int(node.kind) == 0 and len(node.annotations) == 1
                and 'num_stages' in node.annotations
                and self.constant(node.annotations['num_stages']) == 1
                and not isinstance(self.constant(node.annotations['num_stages']),bool)
                and (getattr(node,'step',None) is None or self.constant(node.step)==1)):
            return
        raise Unsupported(f'{path}: 带布局/其他 annotations 的 For 尚未支持')

    def alloc(self, node, path):
        if len(node.strides) or len(node.axis_separators) or int(self.expr(node.elem_offset,path)) != 0:
            raise Unsupported(f"{path}: 局部显式 stride/offset 尚未支持")
        shape = tuple(int(self.expr(s,path)) for s in node.shape)
        self.budget.shape(shape)
        key = self.key(node)
        shared = self.phase_shared if node.scope() in self.shared_scopes else None
        if shared is not None and key in shared:
            self.buffers[key] = shared[key]
            return
        buf = Buffer(np.empty(shape,dtype=dtype(str(node.dtype))),initialized=False,
            symbolic=self.symbolic,budget=self.budget,name=str(node.name))
        self.buffers[key] = buf
        if shared is not None:
            shared[key] = buf
        init = self.initializers.get(key)
        if init is not None:
            # The RISC-V lowering stores at all-zero indices; it does not fill the buffer.
            buf.write((0,) * len(shape), cast(self.expr(init, path+'.initializer'), str(node.dtype)), 0)
            self.event(node, path, action='local_var_init', buffer=buf.name)

    @contextmanager
    def bindings(self, mapping, keys):
        missing = object()
        previous = {key: mapping.get(key, missing) for key in keys}
        try:
            yield
        finally:
            for key, original in previous.items():
                if original is missing: mapping.pop(key, None)
                else: mapping[key] = original

    @contextmanager
    def block_allocations(self, block, path):
        with self.bindings(self.buffers, (self.key(b) for b in block.alloc_buffers)):
            for i,b in enumerate(block.alloc_buffers):
                self.alloc(b, path+f'.alloc_buffers[{i}]')
            yield

    def check_assumption(self, node, path):
        if type(node.value).__name__ != 'StringImm' or str(getattr(node.node,'dtype','')) != 'bool':
            raise Unsupported(f'{path}: tl.assume 需要布尔条件和字符串说明')
        value = self.expr(node.node, path+'.condition')
        if isinstance(value,Expr) or not bool(value):
            raise Unsupported(f'{path}: 输入未能验证 tl.assume 条件')

    def launch_cooperative(self, node, path, extent):
        shared = {}
        lanes = []
        var = node.node.var
        for tid in range(extent):
            lane = copy(self)
            lane.env, lane.buffers = dict(self.env), dict(self.buffers)
            lane.loop_specs, lane.loop_stack = dict(self.loop_specs), list(self.loop_stack)
            lane.env[self.key(var)] = cast(tid, str(var.dtype))
            lane.phase_shared = shared
            lane.shuffle_values = {}
            lane.nonunit_threads += 1
            lanes.append(lane.cooperate(node.body, path+'.body'))
        finished = object()
        incoming = [None]*extent
        region = self.budget.new_region()
        try:
            while True:
                arrivals = []
                for tid,lane in enumerate(lanes):
                    with self.budget.iteration(region,tid):
                        try:
                            arrivals.append(lane.send(incoming[tid]))
                        except StopIteration:
                            arrivals.append(finished)
                if all(x is finished for x in arrivals):
                    return
                if any(x is finished for x in arrivals) or any(x[:2] != arrivals[0][:2] for x in arrivals):
                    raise SemanticFault('物理线程没有全部到达同一重放点', rule_id='SYNC-DIVERGENT-BARRIER')
                kind = arrivals[0][0]
                if kind == 'barrier':
                    if any(x != arrivals[0] for x in arrivals):
                        raise SemanticFault('物理线程屏障作用域不一致', rule_id='SYNC-DIVERGENT-BARRIER')
                    region = self.budget.new_region()
                    incoming = [None]*extent
                else:
                    # A warp exchange is NOT a shared-memory barrier: retain
                    # this phase's access history when resuming the lanes.
                    incoming = []
                    for tid,(_,_,mode,mask,arg,width,mask_bits,_) in enumerate(arrivals):
                        warp0 = tid//32*32
                        other = arrivals[warp0]
                        if mask != other[3] or (mode != 'idx' and arg != other[4]):
                            raise Unsupported('Warp shuffle: 各 lane 的 mask 或 delta 不一致')
                        source = shuffle_source(mode,tid,extent,arg,width,mask,mask_bits)
                        incoming.append(arrivals[source][-1])
        finally:
            for lane in lanes: lane.close()

    def cooperate(self, node, path):
        if self.key(node) not in self.sync_subtrees:
            self.stmt(node,path)
            return
        # Do not hold a Budget context manager over yield: all lanes share it.
        self.budget.tick()
        if self.is_sync(node):
            scope = self.sync_scope(node.value)
            self.event(node,path,action='thread_phase_barrier',scope=scope)
            yield ('barrier',path,scope)
            return
        kind = type(node).__name__
        site = self.shuffle_sites.get(self.key(node))
        if site is not None:
            # Read all lane values before any lane can commit this statement.
            call, width = site
            args = call.args
            with self.budget.at('tvm_node',0,**self.location(call,path+'.value')):
                mask_bits = dtype(str(args[0].dtype)).itemsize*8
                mask = concrete_index(self.expr(args[0],path+'.mask')) & ((1 << mask_bits)-1)
                if mask >> 32:
                    raise Unsupported('Warp shuffle: 参与掩码只能使用低 32 位')
                amount = concrete_index(self.expr(args[2],path+'.lane'))
                mode = SHUFFLES[self.op_name(call)]
                # AsIndex sign-extends signed operands, zero-extends unsigned
                # operands. Negative bit patterns are unsigned for RemUI.
                amount &= 0xffffffff if mode == 'xor' else (1<<64)-1
                if mode in {'up','down'} and amount >= 1<<63:
                    raise Unsupported('Warp shuffle: up/down 需要非负、无溢出的 lane delta')
                value = cast(self.expr(args[1],path+'.payload'),str(call.dtype))
            result = yield ('shuffle',path,mode,mask,amount,width,mask_bits,value)
            self.shuffle_values[self.key(call)] = result
            if kind == 'BufferStore':
                try:
                    self.stmt(node,path)
                finally:
                    self.shuffle_values.pop(self.key(call),None)
                return
        if kind in {'SeqStmt','BlockRealize'}:
            for suffix,child in self.statement_children(node):
                yield from self.cooperate(child,path+suffix)
        elif kind == 'Block':
            with self.block_allocations(node,path):
                yield from self.cooperate(node.body,path+'.body')
        elif kind == 'LetStmt':
            with self.bindings(self.env,[self.key(node.var)]):
                try:
                    self.env[self.key(node.var)] = self.expr(node.value,path+'.value')
                finally:
                    if site is not None:
                        self.shuffle_values.pop(self.key(site[0]),None)
                yield from self.cooperate(node.body,path+'.body')
        elif kind == 'AttrStmt':
            if str(node.attr_key) == 'tl.assume':
                self.check_assumption(node,path)
                yield from self.cooperate(node.body,path+'.body')
            elif self.is_launch_hint(node):
                yield from self.cooperate(node.body,path+'.body')
            else:
                var = node.node.var
                with self.bindings(self.env,[self.key(var)]):
                    self.env[self.key(var)] = cast(0,str(var.dtype))
                    yield from self.cooperate(node.body,path+'.body')
        elif kind == 'For':
            lo,n = self.constant(node.min),self.constant(node.extent)
            with self.bindings(self.env,[self.key(node.loop_var)]):
                for i in range(lo,lo+n):
                    self.budget.tick()
                    self.env[self.key(node.loop_var)] = cast(i,str(node.loop_var.dtype))
                    yield from self.cooperate(node.body,path+'.body')
        else:
            raise Unsupported(f'{path}: 屏障分相没有为 {kind} 建模')

    def uses(self, node, variable):
        # Consider executable expressions, not declarative reads/writes metadata.
        kind = type(node).__name__
        if kind == "Block":
            return self.uses(node.body,variable) or (node.init is not None and self.uses(node.init,variable))
        if kind == "BlockRealize":
            return self.uses(node.block,variable) or self.uses(node.predicate,variable)
        if kind == "AttrStmt": return self.uses(node.body,variable) or self.uses(node.value,variable)
        if kind == "SeqStmt": return any(self.uses(s,variable) for s in node.seq)
        found = []
        self.tvm.tir.stmt_functor.post_order_visit(node,lambda n: found.append(True) if isinstance(n,self.tvm.tir.Var) and n.same_as(variable) else None)
        return bool(found)

    def stmt(self, node, path="body"):
        self.budget.tick()
        try:
            with self.budget.at("tvm_node",0,**self.location(node,path)):
                return self._stmt(node,path)
        except SemanticFault as exc:
            exc.line = None
            exc.details.setdefault("tvm_location",self.location(node,path))
            raise

    def _stmt(self, node, path):
        kind = type(node).__name__
        if kind == "SeqStmt":
            for i,s in enumerate(node.seq): self.stmt(s,path+f".seq[{i}]")
        elif kind == "LetStmt":
            key = self.key(node.var);old = self.env.get(key)
            self.env[key] = self.expr(node.value,path+".value")
            try: self.stmt(node.body,path+".body")
            finally:
                if old is None: self.env.pop(key,None)
                else: self.env[key] = old
        elif kind == "For":
            # Serial, Parallel, Unrolled only; vectorized/thread bindings need rules.
            loop_kind = int(node.kind)
            if loop_kind not in {0,1,3} and not self.vector_copy_loop(node):
                raise Unsupported(f"{path}: For kind={node.kind} 尚未支持")
            self.check_for_metadata(node,path)
            lo,n = int(self.expr(node.min,path+".min")),int(self.expr(node.extent,path+".extent"))
            if n<0: raise Unsupported(f"{path}: negative extent")
            region = self.budget.new_region() if loop_kind in {1,2} else None
            key = self.key(node.loop_var)
            old = self.loop_specs.get(key)
            missing=object()
            old_value=self.env.get(key,missing)
            self.loop_specs[key] = (lo,n,loop_kind)
            self.loop_stack.append(loop_kind)
            try:
                for i in range(lo,lo+n):
                    self.env[key] = cast(i,str(node.loop_var.dtype))
                    if region is None: self.stmt(node.body,path+".body")
                    else:
                        with self.budget.iteration(region,i): self.stmt(node.body,path+".body")
            except LoopBreak:
                if loop_kind!=0: raise
            finally:
                self.loop_stack.pop()
                if old_value is missing: self.env.pop(key,None)
                else: self.env[key]=old_value
                if old is None: self.loop_specs.pop(key,None)
                else: self.loop_specs[key] = old
        elif kind == "IfThenElse":
            cond = self.expr(node.condition,path+".condition")
            if isinstance(cond,Expr): raise Unsupported(f"{path}: 数据相关符号分支尚未支持；可执行具体输入")
            branch = node.then_case if bool(cond) else node.else_case
            if branch is not None: self.stmt(branch,path+(".then" if bool(cond) else ".else"))
        elif kind == "BufferStore":
            if getattr(node,"predicate",None) is not None: raise Unsupported(f"{path}: predicated BufferStore 未支持")
            buf = self.buffers[self.key(node.buffer)]
            idx = tuple(self.expr(i,path+f".indices[{j}]") for j,i in enumerate(node.indices))
            value = self.expr(node.value,path+".value")
            buf.write(idx,value,0)
            if buf.position is not None:
                self.writer_locations[(buf.position,tuple(map(int,idx)))] = self.location(node,path)
            self.event(node,path,action="write",buffer=buf.name,index=[int(i) for i in idx])
        elif kind == "Evaluate":
            if self.is_sync(node):
                scope = self.sync_scope(node.value)
                if self.nonunit_threads:
                    raise Unsupported(f'{path}: 多物理线程屏障必须通过分相执行')
                self.event(node,path,action='serial_barrier',scope=scope)
            elif type(node.value).__name__ == "Call": self.tile(node.value,path+".value")
            elif type(node.value).__name__ == "IntImm" and int(node.value.value) == 0: pass
            else: raise Unsupported(f"{path}: 未支持 Evaluate {type(node.value).__name__}")
        elif kind == "AttrStmt":
            if str(node.attr_key) == 'tl.assume':
                self.check_assumption(node,path)
                self.stmt(node.body,path+'.body')
                return
            if str(node.attr_key)==SWIZZLE_ATTR and self.op_name(node.value)==TIR_TUPLE:
                problem=self.tail_error(node.value,TIR_TUPLE,metadata=True)
                if problem: raise Unsupported(problem)
                self.event(node.value,path+'.value',action='serial_noop',operation=TIR_TUPLE)
                self.stmt(node.body,path+'.body')
                return
            if self.is_launch_hint(node):
                self.stmt(node.body,path+'.body')
                return
            if str(node.attr_key) != "thread_extent":
                raise Unsupported(f"{path}: 未支持 AttrStmt {node.attr_key}")
            var,tag = node.node.var,str(node.node.thread_tag)
            extent = int(self.expr(node.value,path+".value"))
            if extent<1: raise Unsupported(f"{path}: 非法 thread extent")
            if tag.startswith("blockIdx."):
                region = self.budget.new_region() if self.key(node) in self.block_launches_with_sync and extent > 1 else None
                with self.bindings(self.env,[self.key(var)]):
                    for i in range(extent):
                        self.env[self.key(var)] = cast(i,str(var.dtype))
                        with self.budget.iteration(region,i) if region is not None else nullcontext():
                            self.stmt(node.body,path+".body")
            elif self.key(node) in self.cooperative_launches:
                self.launch_cooperative(node,path,extent)
            elif tag.startswith("threadIdx.") and (extent == 1 or not self.uses(node.body,var)):
                self.env[self.key(var)] = cast(0,str(var.dtype))
                self.nonunit_threads += extent!=1
                try: self.stmt(node.body,path+".body")
                finally: self.nonunit_threads -= extent!=1
            else:
                raise Unsupported(f"{path}: 使用了物理线程轴 {tag}；当前对象执行器不猜线程映射")
        elif kind == "BlockRealize":
            if not bool(self.expr(node.predicate,path+".predicate")): return
            block = node.block
            if len(block.iter_vars) != len(node.iter_values): raise Unsupported(f"{path}: Block iter mismatch")
            for var,val in zip(block.iter_vars,node.iter_values):
                self.env[self.key(var.var)] = self.expr(val,path+".iter_value")
            if block.init is None:
                self.stmt(block,path+".block")
            else:
                # Standard unscheduled reduction: a direct, complete serial
                # domain. Scheduled/rfactor/predicated forms need other rules.
                if (type(node.predicate).__name__ != 'IntImm' or not int(node.predicate.value)
                        or len(block.alloc_buffers) or len(block.match_buffers) or len(block.annotations)):
                    raise Unsupported(f'{path}: reduction init 仅支持无谓词、无局部重分配的标准块')
                first = [];axes = [];bound_axes = set()
                for iv,actual in zip(block.iter_vars,node.iter_values):
                    axis_kind = int(iv.iter_type)
                    if axis_kind not in {0,2} or type(actual).__name__ != 'Var' or iv.dom is None:
                        raise Unsupported(f'{path}: reduction init 需要直接的串行归约轴映射')
                    lo = int(self.expr(iv.dom.min,path+'.domain.min'))
                    size = int(self.expr(iv.dom.extent,path+'.domain.extent'))
                    key = self.key(actual)
                    if key in bound_axes:
                        raise Unsupported(f'{path}: 块轴映射必须独立，不能共用同一循环轴')
                    bound_axes.add(key)
                    if axis_kind == 0:
                        spec = self.loop_specs.get(key)
                        if spec is None or spec[:2] != (lo,size):
                            raise Unsupported(f'{path}: 空间轴必须直接对应完整循环域')
                        continue
                    if self.loop_specs.get(self.key(actual)) != (lo,size,0):
                        raise Unsupported(f'{path}: 归约循环必须完整、串行且与 Block 域一致')
                    if self.uses(block.init,iv.var) or self.uses(block.init,actual):
                        raise Unsupported(f'{path}: 初始化依赖归约轴，尚未建立执行契约')
                    first.append(int(self.env[self.key(iv.var)]) == lo)
                    axes.append({'variable':str(iv.var.name),'min':lo,'extent':size})
                if not axes: raise Unsupported(f'{path}: init 块缺少显式归约轴')
                initialize = all(first)
                if initialize:
                    self.stmt(block.init,path+'.block.init')
                    self.reductions.append({**self.location(block,path+'.block'),
                        'kind':'block_init','axes':axes,'initialization':'first_reduction_iteration',
                        'order':'original serial loop nesting'})
                self.stmt(block.body,path+'.block.body')
        elif kind == "Block":
            if node.init is not None:
                raise Unsupported(f"{path}: 独立 Block.init 尚未支持")
            self.check_block_metadata(node,path)
            with self.block_allocations(node,path):
                self.stmt(node.body,path+".body")
        else:
            raise Unsupported(f"{path}: 未支持语句 {kind}")

    def run(self, buffers, symbols=None, symbolic=False):
        self.budget = buffers[0].budget if buffers else Budget(self.max_steps)
        self.symbolic,self.env,self.buffers,self.events,self.reductions = symbolic,{},{},[],[]
        self.writer_locations = {}
        self.loop_specs = {}
        self.loop_stack,self.nonunit_threads = [],0
        self.phase_shared = None
        self.shuffle_values = {}
        if len(self.fun.params) != len(buffers): raise Unsupported("TIR 参数数量与测试契约不同")
        for param,buf in zip(self.fun.params,buffers):
            if param not in self.fun.buffer_map: raise Unsupported("目前只支持 buffer 参数")
            spec = self.fun.buffer_map[param]
            if len(spec.axis_separators): raise Unsupported('带 axis_separators 的 buffer 尚未支持')
            if dtype(str(spec.dtype)) != buf.dtype or len(spec.shape) != len(buf.shape):
                raise Unsupported("TIR buffer dtype/rank 与测试契约不同")
            for dimension,size in zip(spec.shape,buf.shape):
                if type(dimension).__name__ in {"Var","SizeVar"}:
                    key = self.key(dimension)
                    previous = self.env.setdefault(key,cast(size,str(dimension.dtype)))
                    if int(previous) != size: raise Unsupported("共享动态 shape 符号不一致")
            self.buffers[self.key(spec)] = buf
        for param,buf in zip(self.fun.params,buffers):
            spec = self.fun.buffer_map[param]
            if tuple(int(self.expr(s,"buffer.shape")) for s in spec.shape) != buf.shape:
                raise Unsupported("TIR shape 与用例不一致；当前只直接绑定符号维度，不求解复合形状表达式")
            if int(self.expr(spec.elem_offset,"buffer.elem_offset")) != 0:
                raise Unsupported("非零 elem_offset 尚未支持")
            actual = tuple(i//buf.dtype.itemsize for i in buf.values.strides)
            if len(spec.strides) and tuple(int(self.expr(s,"buffer.strides")) for s in spec.strides) != actual:
                raise Unsupported("显式 stride 与用例连续布局不一致")
        self.stmt(self.fun.body)
        return {"steps":self.budget.steps,"backend":"tvm_tir_nodes",
                "execution_model":("serialized lane exchanges and shared barrier phases with checked accesses" if self.shuffle_sites else "serialized shared barrier phases with checked lane accesses"
                                   if self.cooperative_launches else "logical blocks; only unused or unit physical thread axes"),
                "location_model":"node paths + original spans; no exported-text line mapping",
                "node_events":self.events,"reductions":self.reductions}
