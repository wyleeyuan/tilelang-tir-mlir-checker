"""Strict structured-MLIR subset parser/interpreter, independent of TIR frontend.

This is not the official MLIR verifier. Unknown syntax, side-effectful symbolic
branches, calls, layouts and unsupported operations stop execution explicitly.
"""
from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass, field

import numpy as np

from .execution import Buffer, Budget, Expr, SemanticFault, Unsupported, apply, cast, concrete_index, dtype, select, storage_type, trace_value, mlir_cast, mlir_cmpi, mlir_cmpf, scalar_op, scalar_type, while_test
from .capabilities import AFFINE_MATMUL_MAP, MAP_ALIAS, MLIR_BATCH_CASTS, MLIR_BATCH_SCALARS, MLIR_MATMUL, MLIR_OPS
from .tile_ops import (accumulate_matrix,
                       matmul_shape, matrix_step, matrix_transposes, parse_matrix_map, reduction_details)

SSA_NAME = r"%[\w.$-]+"
SSA = rf"{SSA_NAME}(?:#\d+)?"
SCALAR_KIND = r'(?:i(?:1|8|16|32|64)|index|bf16|f(?:16|32|64)|f8E4M3FN)'
SCF_WHILE = (rf'(?:(?P<result>{SSA_NAME})(?::(?P<count>[1-9]\d*))?\s*=\s*)?'
             r'scf\.while\s*(?:\((?P<init>[^()]*)\)\s*)?:\s*'
             rf'\((?P<ins>[^()]*)\)\s*->\s*(?:\((?P<outs>[^()]*)\)|(?P<out>{SCALAR_KIND}))\s*\{{')
SCF_CONDITION = rf'scf\.condition\(\s*({SSA})\s*\)(?:\s+(.*))?'
MLIR_BATCH_PATTERNS = {
    name: re.escape(name) + ' ' + ', '.join([f'({SSA})'] * arity) + r' : (\w+)'
    for name, (_, arity) in MLIR_BATCH_SCALARS.items()
}
LAYOUT_INT = r'(?:\?|-?\d+)'
LAYOUT_STRIDES = rf'(?:{LAYOUT_INT}(?:, {LAYOUT_INT})*)?'
MEMREF = rf'memref<(?:(?:\d+|\?)x)*\w+(?:, strided<\[{LAYOUT_STRIDES}\](?:, offset: {LAYOUT_INT})?>)?>'
VIEW_LIST = rf'(?:\s*(?:{SSA}|-?\d+)\s*(?:,\s*(?:{SSA}|-?\d+)\s*)*)?'
REINTERPRET = (rf'memref\.reinterpret_cast ({SSA}) to offset: \[({VIEW_LIST})\], '
               rf'sizes: \[({VIEW_LIST})\], strides: \[({VIEW_LIST})\] : ({MEMREF}) to ({MEMREF})')

def split_types(text):
    parts=[];depth=0;start=0
    for i,c in enumerate(text):
        if c in '<[(':depth+=1
        elif c in '>])':depth-=1
        elif c==',' and depth==0:parts.append(text[start:i].strip());start=i+1
    parts.append(text[start:].strip())
    return parts


def while_parts(text, pattern):
    parts=tuple(x.strip() for x in text.split(',')) if text.strip() else ()
    if not all(re.fullmatch(pattern,p) for p in parts):
        raise Unsupported('scf.while 只支持标准单行标量参数和 SSA 列表')
    return parts


def while_values(text, kinds):
    values,sep,types=(text or '').partition(':')
    operands=while_parts(values,SSA)
    if (bool(operands)!=bool(sep) or while_parts(types,SCALAR_KIND)!=kinds
            or len(operands)!=len(kinds)):
        raise Unsupported('scf.while 终结符的参数或类型不匹配')
    return operands


def while_text(node):
    header=re.fullmatch(SCF_WHILE,node.text)
    if not header:
        raise Unsupported(f'L{node.line}: 未支持的 scf.while 标头')
    result,count,init,ins,outs,out=header.groups()
    ins,outs=while_parts(ins,SCALAR_KIND),while_parts(outs if outs is not None else out,SCALAR_KIND)
    result_count=int(count or 1) if result else 0
    if result_count!=len(outs):
        raise Unsupported(f'L{node.line}: scf.while 声明的结果数量不匹配')
    result_names=(() if not result else
                  tuple(f'{result}#{i}' for i in range(result_count)) if count else (result,))
    binding=rf'({SSA_NAME})\s*=\s*({SSA})'
    pairs=tuple(re.fullmatch(binding,p).groups() for p in while_parts(init or '',binding))
    before_names=tuple(a for a,_ in pairs)
    if len(set(before_names))!=len(ins) or len(pairs)!=len(ins):
        raise Unsupported(f'L{node.line}: scf.while 初始参数数量或名称不匹配')
    body=node.otherwise
    after_names=()
    if body and body[0].text.startswith('^'):
        block=re.fullmatch(r'\^bb\d+(?:\(([^()]*)\))?:',body[0].text)
        if not block:
            raise Unsupported('scf.while 的 after 块需要标准 ^bbN 参数声明')
        arg=rf'({SSA_NAME})\s*:\s*({SCALAR_KIND})'
        params=tuple(re.fullmatch(arg,p).groups() for p in while_parts(block[1] or '',arg))
        after_names=tuple(a for a,_ in params)
        if tuple(t for _,t in params)!=outs or len(set(after_names))!=len(after_names):
            raise Unsupported('scf.while 的 after 块参数类型或数量不匹配')
        body=body[1:]
    if len(after_names)!=len(outs):
        raise Unsupported('scf.while 缺少 after 块参数')
    if not node.body or not (cond:=re.fullmatch(SCF_CONDITION,node.body[-1].text)):
        raise Unsupported(f'L{node.line}: scf.while 缺少 scf.condition')
    if not body or not (yielded:=re.fullmatch(r'scf\.yield(?:\s+(.*))?',body[-1].text)):
        raise Unsupported(f'L{node.line}: scf.while 缺少 scf.yield')
    condition_values=while_values(cond[2],outs)
    yield_values=while_values(yielded[1],ins)
    for term in (node.body[-1],body[-1]):
        if term.body or term.otherwise:
            raise Unsupported('scf.while 终结符不能有 region')
    return (pairs,ins,outs,result_names,after_names,body,
            cond[1],condition_values,yield_values)

def memref_type(text):
    m=re.fullmatch(rf'memref<((?:(?:\d+|\?)x)*)(\w+)(?:, strided<\[({LAYOUT_STRIDES})\](?:, offset: ({LAYOUT_INT}))?>)?>',text)
    if not m:raise Unsupported('memref 布局格式尚未支持')
    dims=m[1].rstrip('x').split('x') if m[1] else []
    return dims,m[2],m[3],m[4]


def check_view_type(text, buf):
    dims,kind,layout,offset=memref_type(text)
    if (len(dims)!=len(buf.shape) or any(d!='?' and int(d)!=n for d,n in zip(dims,buf.shape))
            or storage_type(dtype(kind))!=storage_type(buf.dtype)):
        raise Unsupported('视图声明 shape/dtype 与存储不匹配')
    if layout is None:
        strides=tuple(math.prod(buf.shape[i+1:]) for i in range(len(dims)))
        # Identity layout derives strides from the actual dynamic dimensions.
    else:
        strides=tuple('?' if x.strip()=='?' else int(x) for x in layout.split(',') if x.strip())
    if (offset!='?' and int(offset or '0')!=buf.element_offset
            or len(strides)!=len(buf.element_strides)
            or any(s not in ('?',v) for s,v in zip(strides,buf.element_strides))):
        raise Unsupported('视图声明 offset/stride 与存储不匹配')
    return kind


def matmul_text(text, aliases):
    op, _, body = text.partition(' ')
    match=re.fullmatch(rf'(.*?)\bins\(({SSA}),\s*({SSA})\s*:\s*({MEMREF}),\s*({MEMREF})\)'
                       rf'\s+outs\(({SSA})\s*:\s*({MEMREF})\)\s*(.*)',body)
    if not match:
        raise Unsupported('matmul 文本只支持两个 ins 和一个 outs memref')
    head,a,b,ak,bk,c,ck,tail=match.groups()
    head,tail=head.strip(),tail.strip()
    attrs=' '.join(piece[1:-1].strip() if piece.startswith('{') and piece.endswith('}') else piece
                   for piece in (head,tail) if piece)
    maps, cast_mode = None,None
    while attrs:
        indexing=re.match(r'indexing_maps\s*=\s*\[([^\[\]]*)\]',attrs)
        conversion=re.match(r'cast\s*=\s*#linalg\.type_fn<(cast_signed|cast_unsigned)>|'
                            r'cast\((cast_signed|cast_unsigned)\)',attrs)
        if indexing and maps is None:
            items=indexing[1].strip()
            entry=re.compile(rf'({AFFINE_MATMUL_MAP}|{MAP_ALIAS})\s*(,|$)')
            result=[]
            while items:
                m=entry.match(items)
                if not m: raise Unsupported('matmul indexing_maps 不是已支持的二维投影')
                if m[1].startswith('#'):
                    if m[1] not in aliases: raise Unsupported('matmul 引用了未知 affine map')
                    result.append(aliases[m[1]])
                else:
                    result.append(parse_matrix_map(m[1]))
                items=items[m.end():].strip()
                if m[4] and not items: raise Unsupported('matmul indexing_maps 含多余逗号')
            maps=tuple(result)
            attrs=attrs[indexing.end():].lstrip()
        elif conversion and cast_mode is None:
            cast_mode=conversion[1] or conversion[2]
            attrs=attrs[conversion.end():].lstrip()
        else:
            raise Unsupported('未知、重复或不支持的 matmul 文本属性')
        if attrs.startswith(','):
            attrs=attrs[1:].strip()
            if not attrs: raise Unsupported('matmul 属性含多余逗号')
    return (a,b,c),(ak,bk,ck),*matrix_transposes(op,maps),cast_mode=='cast_unsigned'



@dataclass
class Node:
    text: str
    line: int
    body: list = field(default_factory=list)
    otherwise: list = field(default_factory=list)


def _parse(source, aliases):
    root = []
    stack = [(None, root)]
    for number, raw in enumerate(source.splitlines(), 1):
        line = raw.split("//", 1)[0].strip()
        if not line:
            continue
        alias=re.fullmatch(rf'({MAP_ALIAS})\s*=\s*({AFFINE_MATMUL_MAP})',line)
        if alias and len(stack)==1 and not root:
            if alias[1] in aliases: raise Unsupported('重复的 matmul affine map 别名')
            aliases[alias[1]]=parse_matrix_map(alias[2])
            continue
        if line in {"} else {", "} do {"}:
            if len(stack)<2:
                raise Unsupported(f"L{number}: 无法解析 {line}")
            header,body=stack[-1]
            matches=(re.fullmatch(SCF_WHILE,header.text) if line=="} do {"
                     else re.match(rf'^(?:{SSA_NAME}\s*=\s*)?scf\.if\s',header.text))
            if not matches or body is header.otherwise:
                raise Unsupported(f"L{number}: 无法解析 {line}")
            node = stack.pop()[0]
            stack.append((node, node.otherwise))
            continue
        if line == "}":
            if len(stack) == 1:
                raise Unsupported("MLIR region 括号不平衡")
            stack.pop()
            continue
        node = Node(line, number)
        stack[-1][1].append(node)
        if line.endswith("{"):
            stack.append((node, node.body))
        elif ("{" in line or "}" in line) and line.split()[0] not in MLIR_MATMUL:
            raise Unsupported(f"L{number}: 不支持的内联 region/属性格式")
    if len(stack) != 1:
        raise Unsupported("MLIR region 未闭合")
    if len(root) != 1 or root[0].text != "module {":
        raise Unsupported("MLIR 执行需要单个无属性 module")
    functions = root[0].body
    if len(functions) != 1 or not functions[0].text.startswith("func.func @"):
        raise Unsupported("MLIR 执行需要单个公开函数且不含外部调用")
    return functions[0]


class MLIRInterpreter:
    OPS = MLIR_OPS

    def __init__(self, source):
        self.map_aliases,self.matmul_specs={},{}
        self.while_specs={}
        self.function = _parse(source,self.map_aliases)
        match = re.fullmatch(r"func\.func @([\w.$-]+)\((.*)\) \{", self.function.text)
        if match is None:
            raise Unsupported("未支持的 MLIR 函数头格式")
        self.symbol = match[1]
        self.parameters = []
        for part in split_types(match[2]) if match[2] else []:
            arg = re.fullmatch(rf"\s*({SSA}): ({MEMREF})\s*", part)
            if arg is None:raise Unsupported("仅支持 memref 参数")
            shape,kind,strides,offset=memref_type(arg[2])
            self.parameters.append((arg[1], shape, kind))
            # Validate known static stride contracts on run(), do not erase layouts.
            if not hasattr(self,'layouts'):self.layouts={}
            self.layouts[arg[1]]=(strides,offset)
        self._preflight(self.function.body)

    def _preflight(self, nodes, *, inside_while=False, direct_while=False):
        for node in nodes:
            if re.fullmatch(SCF_WHILE,node.text):
                spec=self.while_specs[node.line]=while_text(node)
                self._preflight(node.body[:-1],inside_while=True,direct_while=True)
                self._preflight(spec[5][:-1],inside_while=True,direct_while=True)
                continue
            text = re.sub(rf"^{SSA}\s*=\s*", "", node.text)
            token = re.match(r"[\w.]+",text)
            op = token[0] if token else ""
            if (op=='scf.condition' or (inside_while and op=='return')
                    or (direct_while and op in {'scf.yield','scf.reduce'})):
                raise Unsupported(f'L{node.line}: {op} 终结符位置不合法')
            if op not in self.OPS:
                raise Unsupported(f"L{node.line}: 未支持 MLIR 操作 {op}")
            if op in MLIR_MATMUL:
                if node.body or node.otherwise or text!=node.text:
                    raise Unsupported(f'L{node.line}: matmul 只支持无返回值的标准命名操作')
                self.matmul_specs[node.line]=matmul_text(text,self.map_aliases)
                continue
            patterns={
                'scf.for':rf'scf.for {SSA} = {SSA} to {SSA} step {SSA} \{{',
                'scf.parallel':r'scf.parallel \([^)]+\) = \([^)]+\) to \([^)]+\) step \([^)]+\) \{',
                'scf.if':rf'scf.if {SSA}(?: -> \(\w+\))? \{{',
                'scf.yield':rf'scf.yield(?: {SSA} : \w+)?',
                'scf.reduce':r'scf.reduce', 'return':r'return',
                'memref.alloca':rf'memref.alloca\([^)]*\) : {MEMREF}',
                'memref.alloc':rf'memref.alloc\([^)]*\) : {MEMREF}',
                'memref.load':rf'memref.load {SSA}\[[^]]*\] : {MEMREF}',
                'memref.store':rf'memref.store {SSA}, {SSA}\[[^]]*\] : {MEMREF}',
                'memref.dim':rf'memref.dim {SSA}, {SSA} : {MEMREF}',
                'arith.constant':r'arith.constant (?:true|false|[^ ]+ : \w+)',
                'arith.cmpi':rf'arith.cmpi (?:eq|ne|[su](?:lt|le|gt|ge)), {SSA}, {SSA} : \w+',
                'arith.select':rf'arith.select {SSA}, {SSA}, {SSA} : \w+',
                'linalg.fill':rf'linalg.fill ins\({SSA} : \w+\) outs\({SSA} : {MEMREF}\)',
            }
            pattern=patterns.get(op)
            if op=='arith.negf':pattern=rf'arith.negf {SSA} : \w+'
            if op=='memref.copy':pattern=rf'memref.copy {SSA}, {SSA} : {MEMREF} to {MEMREF}'
            if op in {'math.exp','math.exp2','math.absf','math.rsqrt','math.sqrt'}:pattern=rf'math\.\w+ {SSA} : \w+'
            if op=='arith.cmpf':pattern=rf'arith.cmpf \w+, {SSA}, {SSA} : \w+'
            if op=='memref.subview':pattern=rf'memref.subview {SSA}\[[^]]*\] \[[^]]*\] \[[^]]*\] : {MEMREF} to {MEMREF}'
            if op=='memref.reinterpret_cast':pattern=REINTERPRET
            if op in MLIR_BATCH_CASTS | {'arith.index_cast','arith.index_castui','arith.extf','arith.truncf',
                       'arith.sitofp','arith.uitofp','arith.extsi','arith.extui','arith.trunci'}:
                pattern=rf'arith\.\w+ {SSA} : \w+ to \w+'
            if op in {'arith.addi','arith.subi','arith.muli','arith.addf','arith.subf','arith.mulf','arith.divf','arith.floordivsi','arith.remsi'}:
                pattern=rf'arith\.\w+ {SSA}, {SSA} : \w+'
            pattern = MLIR_BATCH_PATTERNS.get(op, pattern)
            if not pattern or not re.fullmatch(pattern,text):
                raise Unsupported(f'L{node.line}: 未支持 {op} 的具体语法')
            for type_name in re.findall(r'\b(?:bf16|f16|f32|f64|i1|i8|i16|i32|i64|index)\b',text):
                dtype(type_name)
            if node.body and op not in {"scf.for", "scf.parallel", "scf.if"}:
                raise Unsupported(f"L{node.line}: 未支持的 region")
            self._preflight(node.body,inside_while=inside_while)
            self._preflight(node.otherwise,inside_while=inside_while)

    def run(self, buffers, symbols=None, symbolic=False):
        if len(buffers) != len(self.parameters):
            raise Unsupported("MLIR 参数数量与测试契约不符")
        self.symbolic = symbolic
        self.budget = buffers[0].budget if buffers else Budget()
        self.pure_only = False
        self.writer_locations, self.reductions = {}, []
        env = {}
        for (param, shape, kind), buf in zip(self.parameters, buffers):
            if len(shape) != len(buf.shape) or any(d != "?" and int(d) != n for d,n in zip(shape,buf.shape)):
                raise Unsupported("MLIR 参数 shape 与测试契约不符")
            if storage_type(dtype(kind)) != storage_type(buf.dtype):
                raise Unsupported("MLIR 参数 dtype 与测试契约不符")
            strides,offset=self.layouts[param]
            if strides is not None:
                strides=[v.strip() for v in strides.split(',')]
                actual=buf.element_strides
                if len(strides)!=len(actual) or any(s!='?' and int(s)!=v for s,v in zip(strides,actual)):
                    raise Unsupported('MLIR 静态 stride 与参数布局不符')
            if offset not in (None,'?','0'):raise Unsupported('非零参数 offset 需要显式 view 契约')
            env[param] = buf
            buf.name = param
        self.sequence(self.function.body, env)
        return {"steps": self.budget.steps, "execution_model": "structured logical loops, serial enumeration of parallel iterations",
                "reductions":self.reductions}

    def sequence(self, nodes, env):
        for node in nodes:
            self.budget.tick()
            with self.budget.at('operation', node.line):
                try:
                    result = self.operation(node, env)
                except SemanticFault as exc:
                    if exc.line is None:
                        exc.line = node.line
                    raise
            if isinstance(result, tuple) and result[0] == "yield":
                return result[1]
        return None

    def value(self, token, env):
        token = token.strip()
        if token not in env:
            raise Unsupported(f"SSA 值未定义：{token}")
        return env[token]

    def indices(self, text, env):
        return tuple(self.value(v, env) for v in text.split(",")) if text.strip() else ()

    def write(self,buf,index,value,line):
        buf.write(index,value,line)
        if buf.position is not None:
            self.writer_locations[(buf.position,buf.root_index(index))]={'stage_text_line':line}

    def execute_while(self,node,env,spec):
        pairs,ins,outs,result_names,after_names,body,test,condition_values,yield_values=spec
        def scalars(names,kinds,local):
            values=tuple(self.value(v,local) for v in names)
            if any(isinstance(v,Buffer) or dtype(scalar_type(v))!=dtype(t)
                   for v,t in zip(values,kinds)):
                raise Unsupported(f'L{node.line}: scf.while 实际值与标量类型不符')
            return values
        carried=scalars((init for _,init in pairs),ins,env)
        while True:
            before=dict(env)
            before.update(zip((name for name,_ in pairs),carried))
            self.sequence(node.body[:-1],before)
            self.budget.tick()  # scf.condition (even a no-value empty loop is bounded)
            take_after=while_test(self.value(test,before))
            forwarded=scalars(condition_values,outs,before)
            if not take_after:
                env.update(zip(result_names,forwarded))
                return
            after=dict(env)
            after.update(zip(after_names,forwarded))
            self.sequence(body[:-1],after)
            self.budget.tick()  # scf.yield
            carried=scalars(yield_values,ins,after)

    def operation(self, node, env):
        original, line = node.text, node.line
        if line in self.while_specs:
            return self.execute_while(node,env,self.while_specs[line])
        assignment = re.match(rf"^({SSA})\s*=\s*(.*)", original)
        target = assignment[1] if assignment else None
        text = assignment[2] if assignment else original
        if text.startswith("scf.for "):
            m = re.fullmatch(rf"scf.for ({SSA}) = ({SSA}) to ({SSA}) step ({SSA}) \{{", text)
            if not m:
                raise Unsupported(f"L{line}: 未支持的 scf.for 格式")
            lower, upper, step = [int(self.value(m[i],env)) for i in (2,3,4)]
            if step <= 0:
                raise Unsupported("scf.for 需要正步长")
            for value in range(lower, upper, step):
                local = dict(env)
                local[m[1]] = np.int64(value)
                with self.budget.at('loop', line, variable=m[1], value=value, lower=lower, upper=upper, step=step):
                    self.sequence(node.body, local)
            return None
        if text.startswith("scf.parallel "):
            m = re.fullmatch(r"scf.parallel \(([^)]+)\) = \(([^)]+)\) to \(([^)]+)\) step \(([^)]+)\) \{", text)
            if not m:
                raise Unsupported(f"L{line}: 未支持的 scf.parallel")
            names = [v.strip() for v in m[1].split(",")]
            bounds = [self.indices(m[i],env) for i in (2,3,4)]
            if any(len(b) != len(names) for b in bounds) or any(int(s) <= 0 for s in bounds[2]):
                raise Unsupported("parallel 循环维度或步长错误")
            region=self.budget.new_region()
            for index in itertools.product(*[range(int(a),int(b),int(s)) for a,b,s in zip(*bounds)]):
                local = dict(env)
                local.update({k:np.int64(v) for k,v in zip(names,index)})
                with self.budget.iteration(region,index):
                    with self.budget.at('parallel', line, variables=names, values=list(index)):
                        self.sequence(node.body,local)
            return None
        if text.startswith("scf.if "):
            m = re.fullmatch(rf"scf.if ({SSA})(?: -> \(([\w]+)\))? \{{", text)
            if not m:
                raise Unsupported(f"L{line}: 未支持的 scf.if")
            cond = self.value(m[1],env)
            if isinstance(cond, Expr):
                if not target or not node.otherwise:
                    raise Unsupported("输入相关符号 if 必须为无副作用标量选择")
                old = self.pure_only
                self.pure_only = True
                try:
                    yes = self.sequence(node.body,dict(env))
                    no = self.sequence(node.otherwise,dict(env))
                finally:
                    self.pure_only = old
                value = select(cond,yes,no)
            else:
                with self.budget.at('guard', line, condition=m[1], value=bool(cond)):
                    value = self.sequence(node.body if bool(cond) else node.otherwise,dict(env))
            if target:
                if value is None:
                    raise Unsupported("if 缺少标量 yield")
                env[target] = value
            return None
        if text in {"scf.yield", "scf.reduce", "return"}:
            return None
        if text.startswith("scf.yield "):
            m = re.fullmatch(rf"scf.yield ({SSA}) : [\w]+",text)
            if not m:
                raise Unsupported("只支持单值 yield")
            return ("yield",self.value(m[1],env))
        if text.startswith("memref.store "):
            if self.pure_only:
                raise Unsupported("符号 if 分支中含 store")
            m = re.fullmatch(rf"memref.store ({SSA}), ({SSA})\[([^]]*)\] : {MEMREF}",text)
            if not m:
                raise Unsupported(f"L{line}: store 格式未支持")
            self.write(self.value(m[2],env),self.indices(m[3],env),self.value(m[1],env),line)
            return None
        if text.startswith('memref.copy '):
            if self.pure_only:raise Unsupported('符号分支含 memref.copy')
            m=re.fullmatch(rf'memref.copy ({SSA}), ({SSA}) : {MEMREF} to {MEMREF}',text)
            source,target=[self.value(m[i],env) for i in (1,2)]
            if source.shape!=target.shape or source.dtype!=target.dtype:raise Unsupported('memref.copy 的 shape/dtype 不同')
            if np.shares_memory(source.values,target.values):raise Unsupported('memref.copy 重叠存储尚未支持')
            for idx in np.ndindex(source.shape):target.write(idx,source.read(idx,line),line)
            return None
        if text.startswith("linalg.fill "):
            if self.pure_only:
                raise Unsupported("符号分支含 fill")
            m = re.fullmatch(rf"linalg.fill ins\(({SSA}) : [\w]+\) outs\(({SSA}) : memref<[^>]+>\)",text)
            if not m:
                raise Unsupported("仅支持标量 linalg.fill")
            buf = self.value(m[2],env)
            for index in np.ndindex(buf.shape):
                buf.write(index,self.value(m[1],env),line)
            return None
        if line in self.matmul_specs:
            if self.pure_only: raise Unsupported('符号分支含 matmul 写入')
            names,types,ta,tb,unsigned=self.matmul_specs[line]
            a,b,c=(self.value(v,env) for v in names)
            shape=matmul_shape(a,b,c,ta,tb)
            kinds=tuple(check_view_type(t,b) for t,b in zip(types,(a,b,c)))
            step=matrix_step(a,b,c,unsigned=unsigned,kinds=kinds)
            self.reductions.append({'line':line,**reduction_details(c,ta,tb,False)})
            accumulate_matrix(a,b,c,ta,tb,shape,line,step,self.write)
            return None
        if target is None:
            raise Unsupported(f"L{line}: 未支持的无返回值操作 {text}")
        if text.startswith(("memref.alloca", "memref.alloc(")):
            if self.pure_only:
                raise Unsupported("符号分支含 allocation")
            m = re.fullmatch(r"memref.alloca?\(([^)]*)\) : memref<((?:(?:\d+|\?)x)*)([\w]+)>",text)
            if not m:
                raise Unsupported("allocation 格式未支持")
            parts = m[2].rstrip("x").split("x") if m[2] else []
            supplied=self.indices(m[1],env)
            if len(supplied)!=parts.count('?'):raise Unsupported('allocation 动态尺寸数量不匹配')
            dynamic = iter(supplied)
            shape = tuple(int(next(dynamic)) if v == "?" else int(v) for v in parts)
            self.budget.shape(shape)
            value = Buffer(np.empty(shape,dtype=dtype(m[3])),initialized=False,
                           symbolic=self.symbolic,budget=self.budget,name=target,allocation_line=line)
        elif text.startswith("memref.load "):
            m = re.fullmatch(rf"memref.load ({SSA})\[([^]]*)\] : ({MEMREF})",text)
            if not m:
                raise Unsupported("load 格式未支持")
            value = cast(self.value(m[1],env).read(self.indices(m[2],env),line),memref_type(m[3])[1])
        elif text.startswith("memref.dim "):
            m = re.fullmatch(rf"memref.dim ({SSA}), ({SSA}) : {MEMREF}",text)
            if not m:
                raise Unsupported("dim 格式未支持")
            value = np.int64(self.value(m[1],env).shape[int(self.value(m[2],env))])
        elif text.startswith('memref.subview '):
            m=re.fullmatch(rf'memref.subview ({SSA})\[([^]]*)\] \[([^]]*)\] \[([^]]*)\] : ({MEMREF}) to ({MEMREF})',text)
            if not m:raise Unsupported(f'L{line}: subview 语法未支持')
            base=self.value(m[1],env)
            def nums(t):return tuple(int(self.value(v.strip(),env)) if v.strip().startswith('%') else int(v) for v in t.split(','))
            offsets,sizes,strides=[nums(m[i]) for i in (2,3,4)]
            dims,*_=memref_type(m[6])
            # Rank-reducing subview must have one unambiguous size-1 projection.
            if len(dims)>len(sizes):raise Unsupported('subview 目标 rank 大于输入 rank')
            choices=[]
            for drop in itertools.combinations([i for i,n in enumerate(sizes) if n==1],len(sizes)-len(dims)):
                kept=[n for i,n in enumerate(sizes) if i not in drop]
                if all(d=='?' or int(d)==n for d,n in zip(dims,kept)):choices.append(drop)
            if len(choices)!=1:raise Unsupported('subview 降维映射不唯一')
            value=base.subview(offsets,sizes,strides,choices[0])
            check_view_type(m[6],value)
        elif text.startswith('memref.reinterpret_cast '):
            m=re.fullmatch(REINTERPRET,text)
            base=self.value(m[1],env)
            if not isinstance(base,Buffer): raise Unsupported('reinterpret_cast 源不是 memref')
            check_view_type(m[5],base)

            def numbers(tokens):
                return tuple(concrete_index(self.value(v,env)) if v.startswith('%') else int(v)
                             for v in map(str.strip,tokens.split(',')) if v)
            offsets,sizes,strides=map(numbers,m.groups()[1:4])
            if len(offsets)!=1: raise Unsupported('reinterpret_cast 需要一个根存储 offset')
            kind=memref_type(m[6])[1]
            value=base.reinterpret_cast(offsets[0],sizes,strides,element_type=kind,line=line)
            check_view_type(m[6],value)
        elif text.startswith('arith.negf '):
            m=re.fullmatch(rf'arith.negf ({SSA}) : (\w+)',text)
            value=apply('neg',[self.value(m[1],env)],m[2])
        elif text.split()[0] in MLIR_BATCH_PATTERNS:
            op=text.split()[0]
            operands=re.fullmatch(MLIR_BATCH_PATTERNS[op],text).groups()
            kind=operands[-1]
            args=[self.value(a,env) for a in operands[:-1]]
            if any(storage_type(dtype(scalar_type(a))) != storage_type(dtype(kind)) for a in args):
                raise Unsupported(f'{op} 的操作数与声明类型不同')
            value=scalar_op(MLIR_BATCH_SCALARS[op][0],args,kind,[kind]*len(args))
        elif text.startswith('math.'):
            m=re.fullmatch(rf'math\.(exp|exp2|absf|rsqrt|sqrt) ({SSA}) : (\w+)',text)
            if not m:raise Unsupported('math 操作属性尚未支持')
            value=apply('abs' if m[1]=='absf' else m[1],[self.value(m[2],env)],m[3])
        elif text.startswith('arith.cmpf '):
            m=re.fullmatch(rf'arith.cmpf (\w+), ({SSA}), ({SSA}) : (\w+)',text)
            if not m:raise Unsupported('cmpf 语法未支持')
            value=mlir_cmpf(m[1],self.value(m[2],env),self.value(m[3],env))
        elif text.startswith("arith.constant "):
            m = re.fullmatch(r"arith.constant (.+?)(?: : (\w+))?",text)
            token,kind = m[1],m[2]
            if token in {"true","false"}:
                value = np.bool_(token == "true")
            elif kind:
                if token.startswith("0x") and (dtype(kind).kind == "f" or dtype(kind).name in {"bfloat16", "float8_e4m3fn"}):
                    value = np.array(int(token,16),dtype="uint"+str(dtype(kind).itemsize*8)).view(dtype(kind))[()]
                else:
                    try:
                        value = np.array(float(token) if dtype(kind).name in {"bfloat16", "float8_e4m3fn"} else token,dtype=dtype(kind))[()]
                    except (TypeError,ValueError,OverflowError) as exc:
                        raise Unsupported(f'L{line}: constant 位模式或文本格式未支持：{token}') from exc
            else:
                raise Unsupported("constant 缺少支持的类型")
        elif text.split(' ',1)[0] in MLIR_BATCH_CASTS | {
                'arith.index_cast','arith.index_castui','arith.extf','arith.truncf',
                'arith.sitofp','arith.uitofp','arith.extsi','arith.extui','arith.trunci'}:
            m = re.fullmatch(rf"arith\.(\w+) ({SSA}) : (\w+) to (\w+)",text)
            if not m:
                raise Unsupported("cast 格式未支持")
            operand = self.value(m[2],env)
            if 'arith.'+m[1] in MLIR_BATCH_CASTS and storage_type(dtype(scalar_type(operand))) != storage_type(dtype(m[3])):
                raise Unsupported(f'arith.{m[1]} 的操作数与声明类型不同')
            value = mlir_cast(m[1],operand,m[3],m[4])
        elif text.startswith("arith.cmpi "):
            m = re.fullmatch(rf"arith.cmpi (\w+), ({SSA}), ({SSA}) : (\w+)",text)
            if not m:
                raise Unsupported("cmpi 格式未支持")
            value = mlir_cmpi(m[1],self.value(m[2],env),self.value(m[3],env),m[4])
        elif text.startswith("arith.select "):
            m = re.fullmatch(rf"arith.select ({SSA}), ({SSA}), ({SSA}) : (\w+)",text)
            if not m:
                raise Unsupported("select 格式未支持")
            value = select(*[self.value(m[i],env) for i in (1,2,3)])
        else:
            m = re.fullmatch(rf"arith\.(addi|subi|muli|addf|subf|mulf|divf|floordivsi|remsi) ({SSA}), ({SSA}) : (\w+)",text)
            if not m:
                raise Unsupported(f"L{line}: 未支持格式 {text}")
            op = {"addi":"add","subi":"sub","muli":"mul","addf":"add","subf":"sub","mulf":"mul","divf":"div","floordivsi":"floordiv","remsi":"rem"}[m[1]]
            value = apply(op,[self.value(m[2],env),self.value(m[3],env)],m[4])
        env[target] = trace_value(value, target, line, text.split()[0])
        return None
