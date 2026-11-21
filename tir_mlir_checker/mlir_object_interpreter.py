"""Bounded execution of official MLIR objects; never parses printed assembly."""
from __future__ import annotations

import itertools
import re
import numpy as np

from .capabilities import MLIR_BATCH_CASTS, MLIR_BATCH_SCALARS, MLIR_MATMUL
from .execution import (Buffer,Budget,Expr,SemanticFault,Unsupported,apply,cast,dtype,
                        select,storage_type,trace_value,mlir_cast,mlir_cmpi,mlir_cmpf,
                        MLIR_CMPF_PREDICATES,scalar_op,concrete_index,while_test)
from .tile_ops import (accumulate_matrix, matmul_shape, matrix_transposes,
                       parse_matrix_map, reduction_details)


class MLIRObjectInterpreter:
    binary = {'arith.addi':'add','arith.subi':'sub','arith.muli':'mul',
              'arith.addf':'add','arith.subf':'sub','arith.mulf':'mul','arith.divf':'div',
              'arith.floordivsi':'floordiv','arith.remsi':'rem'}
    unary = {'arith.negf':'neg','math.exp':'exp','math.exp2':'exp2','math.rsqrt':'rsqrt',
             'math.sqrt':'sqrt','math.log':'log','math.absf':'abs','math.tanh':'tanh'}
    casts = MLIR_BATCH_CASTS | {'arith.index_cast','arith.index_castui','arith.extf','arith.truncf',
             'arith.sitofp','arith.uitofp','arith.extsi','arith.extui','arith.trunci'}

    def __init__(self,loaded,source,*,max_steps=2_000_000):
        import mlir.ir as ir
        self.ir,self.context,self.module,self.source,self.max_steps = ir,*loaded,source,max_steps
        self.lines = source.splitlines()
        # Explicit locations may refer to another file, or to stale source lines.
        self.text_locations = re.search(r'\bloc\s*\(',source) is None
        if len(self.module.operation.attributes):
            raise Unsupported('module 属性（含目标/data layout）尚未建立模拟契约')
        children = list(self.module.body.operations)
        if len(children)!=1 or children[0].operation.name!='func.func':
            raise Unsupported('官方 MLIR 模拟需要单个 func.func；不猜入口或外部调用')
        self.function = children[0].operation
        if len(self.function.regions)!=1 or len(self.function.regions[0].blocks)!=1:
            raise Unsupported('需要单个函数入口 block')
        self.paths, self.matmul_specs = {}, {}
        allowed = set(self.binary)|set(self.unary)|self.casts|set(MLIR_BATCH_SCALARS)|MLIR_MATMUL|{
            'func.func','func.return','arith.constant','arith.cmpi','arith.cmpf','arith.select',
            'memref.alloc','memref.alloca','memref.load','memref.store','memref.dim',
            'memref.copy','memref.subview','memref.reinterpret_cast','scf.for','scf.parallel','scf.if','scf.yield',
            'scf.while','scf.condition','scf.reduce','linalg.fill','linalg.yield'}
        with self.context:
            def inspect(op,path,parent=None,region=None):
                op=op.operation;self.paths[op]=path
                if op.name not in allowed: raise Unsupported(f'{path}: 未支持官方 MLIR 操作 {op.name}（包括未执行分支）')
                attrs={a if isinstance(a,str) else a.name for a in op.attributes}
                defaults={'overflowFlags':'#arith.overflow<none>','fastmath':'#arith.fastmath<none>'}
                for key,value in defaults.items():
                    if key in attrs and str(op.attributes[key])==value: attrs.remove(key)
                segments={'operandSegmentSizes','operand_segment_sizes'}
                view_attrs=segments|{'static_offsets','static_sizes','static_strides'}
                permitted={'func.func':{'sym_name','function_type','sym_visibility'},
                    'arith.constant':{'value'},'arith.cmpi':{'predicate'},'arith.cmpf':{'predicate'},
                    'scf.parallel':{'operandSegmentSizes','operand_segment_sizes'},
                    'linalg.fill':{'operandSegmentSizes','operand_segment_sizes'},
                    'memref.alloc':{'operandSegmentSizes','operand_segment_sizes','alignment'},
                    'memref.alloca':{'operandSegmentSizes','operand_segment_sizes','alignment'},
                    'memref.subview':view_attrs,'memref.reinterpret_cast':view_attrs}.get(op.name,set())
                if op.name in MLIR_MATMUL: permitted=segments|{'indexing_maps','cast'}
                if attrs-permitted:
                    raise Unsupported(f'{path}: {op.name} 未支持属性 {sorted(attrs-permitted)}')
                if op.name=='scf.parallel' and (len(op.results) or len(op.operands)!=3*len(op.regions[0].blocks[0].arguments)):
                    raise Unsupported(f'{path}: 带归约结果的 scf.parallel 尚未支持')
                if op.name=='scf.reduce' and (len(op.operands) or len(op.regions)):
                    raise Unsupported(f'{path}: 非空 scf.reduce 尚未支持')
                if op.name=='scf.while':
                    self.inspect_while(op,path)
                if op.name=='scf.condition' and (parent,region)!=('scf.while',0):
                    raise Unsupported(f'{path}: scf.condition 只允许在 scf.while 的 before 区域')
                if op.name in MLIR_MATMUL:
                    self.matmul_specs[op]=self.inspect_matmul(op,path)
                for ri,r in enumerate(op.regions):
                    if len(r.blocks)>1: raise Unsupported(f'{path}: 多 block/CFG 尚未支持')
                    for bi,b in enumerate(r.blocks):
                        for oi,child in enumerate(b.operations):
                            inspect(child,f'{path}.r{ri}.b{bi}.op{oi}',op.name,ri)
            inspect(self.function,'function')

    def location(self,op):
        loc=str(op.location);line=None
        try: parsed_file,parsed_line=op.location.filename,int(op.location.start_line)
        except (ValueError,AttributeError): parsed_file,parsed_line=None,None
        if self.text_locations and parsed_file=='-':
            candidate=parsed_line
            if 1<=candidate<=len(self.lines) and op.name in self.lines[candidate-1]: line=candidate
        return {'node_path':self.paths.get(op,'module'),'ir_location':loc,
                'stage_text_line':line,'line_mapping':'parsed_text_checked' if line else 'unavailable'}

    def kind(self,type_):
        # Type strings select scalar dtypes only; no operation/region text parsing.
        kind=str(type_)
        if kind=='i1': return 'bool'
        dtype(kind)
        return kind

    def memtype(self,type_):
        try: spec=self.ir.MemRefType(type_)
        except ValueError as exc: raise Unsupported('仅支持 ranked memref，tensor/vector 尚未支持') from exc
        if spec.memory_space is not None and str(spec.memory_space) not in {'0','0 : i64','0 : i32'}:
            raise Unsupported('非默认 memref memory_space 尚未支持')
        self.kind(spec.element_type)
        return spec

    def layout(self,spec,buf):
        try:
            strides,offset=spec.get_strides_and_offset()
        except ValueError as exc:
            raise Unsupported('暂不支持非 strided memref 布局') from exc
        dynamic=self.ir.ShapedType.get_dynamic_stride_or_offset()
        actual=buf.element_strides
        if offset not in (0,dynamic): raise Unsupported('非零参数/局部 offset 需显式 view 契约')
        if len(strides)!=len(actual) or any(s!=dynamic and s!=v for s,v in zip(strides,actual)):
            raise Unsupported('memref 声明 stride 与实际存储不匹配')

    def view_layout(self,spec,buf):
        if (len(spec.shape)!=len(buf.shape)
                or any(d>=0 and d!=n for d,n in zip(spec.shape,buf.shape))
                or storage_type(dtype(self.kind(spec.element_type)))!=storage_type(buf.dtype)):
            raise Unsupported('视图声明 shape/dtype 与实际存储不匹配')
        try:
            strides,offset=spec.get_strides_and_offset()
        except ValueError as exc:
            raise Unsupported('暂不支持非 strided 视图布局') from exc
        dynamic=self.ir.ShapedType.get_dynamic_stride_or_offset()
        if (offset not in (dynamic,buf.element_offset) or len(strides)!=len(buf.element_strides)
                or any(s not in (dynamic,v) for s,v in zip(strides,buf.element_strides))):
            raise Unsupported('视图声明 offset/stride 与实际存储不匹配')

    def decode_view_parameters(self,op,operands):
        sentinel=self.ir.ShapedType.get_dynamic_stride_or_offset()
        keys=('static_offsets','static_sizes','static_strides')
        if any(k not in op.attributes for k in keys):
            raise Unsupported('缺少 memref 视图静态参数')
        arrays=[tuple(int(v) for v in op.attributes[k]) for k in keys]
        values=operands[1:]
        counts=[a.count(sentinel) for a in arrays]
        if sum(counts)!=len(values):
            raise Unsupported('memref 视图动态参数数量不匹配')
        dynamic=iter(values)
        return tuple(tuple(concrete_index(next(dynamic)) if v==sentinel else v for v in a) for a in arrays)

    def inspect_matmul(self,op,path):
        if len(op.operands)!=3 or len(op.results) or any(len(self.memtype(v.type).shape)!=2 for v in op.operands):
            raise Unsupported(f'{path}: matmul 只支持 3 个二维 memref 和无 SSA 结果')
        maps=None
        if 'indexing_maps' in op.attributes:
            maps=tuple(parse_matrix_map(str(self.ir.AffineMapAttr(m))) for m in self.ir.ArrayAttr(op.attributes['indexing_maps']))
        ta,tb=matrix_transposes(op.name,maps)
        if 'cast' in op.attributes and str(op.attributes['cast']) not in (
                '#linalg.type_fn<cast_signed>','#linalg.type_fn<cast_unsigned>'):
            raise Unsupported(f'{path}: 未知的 linalg matmul cast')
        if len(op.regions)!=1 or len(op.regions[0].blocks)!=1:
            raise Unsupported(f'{path}: matmul 需要单 block 标量 body')
        block=op.regions[0].blocks[0]
        kinds=[self.kind(self.memtype(v.type).element_type) for v in op.operands]
        if [self.kind(v.type) for v in block.arguments]!=kinds:
            raise Unsupported(f'{path}: matmul body 的 A/B/C 参数类型不匹配')
        ops=[v.operation for v in block.operations]
        pure=set(self.binary)|set(self.unary)|self.casts|set(MLIR_BATCH_SCALARS)|{
            'arith.constant','arith.cmpi','arith.cmpf','arith.select'}
        if (not ops or ops[-1].name!='linalg.yield' or len(ops[-1].operands)!=1
                or self.kind(ops[-1].operands[0].type)!=kinds[-1]
                or any(v.name not in pure or len(v.regions) for v in ops[:-1])):
            raise Unsupported(f'{path}: matmul body 只支持纯标量指令及一个 C 类型的 yield')
        return ta,tb,block,kinds

    def inspect_while(self,op,path):
        # SCF admits different types/counts on the two edges of the cycle.
        # before operands == initial values; after operands == loop results.
        if len(op.regions)!=2 or any(len(r.blocks)!=1 for r in op.regions):
            raise Unsupported(f'{path}: scf.while 需要各含一个 block 的 before/after 区域')
        before,after=(r.blocks[0] for r in op.regions)
        def terminator(block,name):
            ops=[v.operation for v in block.operations]
            if not ops or ops[-1].name!=name:
                raise Unsupported(f'{path}: scf.while 缺少 {name}')
            return ops[-1]
        condition=terminator(before,'scf.condition')
        yielded=terminator(after,'scf.yield')
        if not len(condition.operands) or str(condition.operands[0].type)!='i1':
            raise Unsupported(f'{path}: scf.condition 需要 i1')
        def types(values): return tuple(v.type for v in values)
        if (types(op.operands)!=types(before.arguments) or types(yielded.operands)!=types(op.operands)
                or types(op.results)!=types(after.arguments)
                or types(tuple(condition.operands)[1:])!=types(op.results)):
            raise Unsupported(f'{path}: scf.while 两条传值路径的参数或结果不匹配')
        for v in itertools.chain(op.operands,op.results):
            if isinstance(v.type,self.ir.MemRefType):
                self.memtype(v.type)
            else:
                self.kind(v.type)

    def value(self,value,env):
        if value not in env: raise Unsupported('官方 SSA 值未绑定')
        return env[value]

    def sequence(self,block,env):
        for view in block.operations:
            op=view.operation;self.budget.tick()
            location=self.location(op);line=location['stage_text_line'] or 0
            try:
                with self.budget.at('mlir_operation',location['stage_text_line'],node_path=location['node_path'],operation=op.name):
                    result=self.operation(op,env,line)
            except SemanticFault as exc:
                if exc.line==0:
                    exc.line=None;exc.args=(str(exc).removeprefix('L0: '),)
                if exc.line is None:
                    exc.line=location['stage_text_line']
                exc.details.setdefault('mlir_location',location)
                raise
            except Unsupported as exc:
                if not getattr(exc,'mlir_location',None): exc.mlir_location=location
                raise
            if result is not None: return result
        return None

    def write(self,buf,index,value,op,line):
        if self.pure_only: raise Unsupported('数据相关符号分支含内存写；仅执行具体输入')
        buf.write(index,value,line)
        if buf.position is not None: self.writer_locations[(buf.position,buf.root_index(index))]=self.location(op)

    def operation(self,op,env,line):
        name=op.name
        operands=[self.value(a,env) for a in op.operands]
        if name in {'scf.condition','scf.yield','func.return','linalg.yield','scf.reduce'}:
            return (name,tuple(operands))
        if name=='scf.for':
            lo,hi,step=map(int,operands[:3]);carried=tuple(operands[3:])
            if step<=0: raise Unsupported('scf.for 需要正步长')
            block=op.regions[0].blocks[0]
            for i in range(lo,hi,step):
                local=dict(env);local.update(zip(block.arguments,(np.int64(i),*carried)))
                yielded=self.sequence(block,local)
                if yielded is None or yielded[0]!='scf.yield': raise Unsupported('scf.for 缺少 yield')
                carried=yielded[1]
            values=carried
        elif name=='scf.while':
            before,after=(r.blocks[0] for r in op.regions)
            carried=operands
            while True:
                local=dict(env)
                local.update(zip(before.arguments,carried))
                # sequence charges the common Budget, including the terminator
                # itself. Even an otherwise empty cycle cannot run unbounded.
                condition=self.sequence(before,local)
                if condition is None or condition[0]!='scf.condition':
                    raise Unsupported('scf.while 缺少 scf.condition')
                test,*forwarded=condition[1]
                if not while_test(test):
                    values=forwarded
                    break
                local=dict(env)
                local.update(zip(after.arguments,forwarded))
                yielded=self.sequence(after,local)
                if yielded is None or yielded[0]!='scf.yield':
                    raise Unsupported('scf.while 缺少 scf.yield')
                carried=yielded[1]
        elif name=='scf.parallel':
            block=op.regions[0].blocks[0];n=len(block.arguments)
            lo,hi,step=operands[:n],operands[n:2*n],operands[2*n:]
            if any(int(s)<=0 for s in step): raise Unsupported('scf.parallel 需要正步长')
            region=self.budget.new_region()
            for index in itertools.product(*[range(int(a),int(b),int(s)) for a,b,s in zip(lo,hi,step)]):
                local=dict(env);local.update(zip(block.arguments,map(np.int64,index)))
                with self.budget.iteration(region,index): self.sequence(block,local)
            return None
        elif name=='scf.if':
            cond=operands[0]
            def branch(i):
                r=op.regions[i]
                if not len(r.blocks): return ()
                yielded=self.sequence(r.blocks[0],dict(env))
                if yielded is None or yielded[0]!='scf.yield': raise Unsupported('scf.if 缺少 yield')
                return yielded[1]
            if isinstance(cond,Expr):
                if not len(op.results) or not len(op.regions[1].blocks): raise Unsupported('数据相关符号 if 需要有值的纯分支')
                old=self.pure_only;self.pure_only=True
                try: yes,no=branch(0),branch(1)
                finally: self.pure_only=old
                values=tuple(select(cond,a,b) for a,b in zip(yes,no))
            else: values=branch(0 if bool(cond) else 1)
        elif name=='memref.store':
            self.write(operands[1],tuple(operands[2:]),operands[0],op,line);return None
        elif name=='memref.copy':
            source,target=operands
            if source.shape!=target.shape or source.dtype!=target.dtype or np.shares_memory(source.values,target.values):
                raise Unsupported('memref.copy shape/dtype 不同或源目标重叠')
            for idx in np.ndindex(source.shape): self.write(target,idx,source.read(idx,line),op,line)
            return None
        elif name=='linalg.fill':
            if len(operands)!=2 or len(op.results): raise Unsupported('仅支持 buffer 形式 linalg.fill')
            block=op.regions[0].blocks[0];body=list(block.operations)
            if len(body)!=1 or body[0].operation.name!='linalg.yield' or list(body[0].operation.operands)!=[block.arguments[0]]:
                raise Unsupported('linalg.fill 的 region 不符合标准标量填充')
            value,buf=operands
            for idx in np.ndindex(buf.shape): self.write(buf,idx,value,op,line)
            return None
        elif name in MLIR_MATMUL:
            if self.pure_only: raise Unsupported('数据相关符号分支含 matmul 写入')
            a,b,c=operands
            ta,tb,block,kinds=self.matmul_specs[op]
            shape=matmul_shape(a,b,c,ta,tb)
            self.reductions.append({**self.location(op),**reduction_details(c,ta,tb,False)})

            def step(av,bv,acc):
                local=dict(env)
                local.update(zip(block.arguments,(cast(v,k) for v,k in zip((av,bv,acc),kinds))))
                return self.sequence(block,local)[1][0]

            accumulate_matrix(a,b,c,ta,tb,shape,line,step,lambda b,i,v,l:self.write(b,i,v,op,l))
            return None
        elif name in {'memref.alloc','memref.alloca'}:
            if self.pure_only: raise Unsupported('数据相关符号分支含 allocation')
            spec=self.memtype(op.results[0].type);supplied=iter(operands)
            shape=tuple(int(next(supplied)) if n<0 else int(n) for n in spec.shape)
            if next(supplied,None) is not None: raise Unsupported('allocation affine symbols 尚未支持')
            self.budget.shape(shape)
            buf=Buffer(np.empty(shape,dtype=dtype(self.kind(spec.element_type))),initialized=False,
                symbolic=self.symbolic,budget=self.budget,name=self.paths[op],allocation_line=line or None)
            self.layout(spec,buf);values=(buf,)
        elif name=='memref.load':
            values=(cast(operands[0].read(tuple(operands[1:]),line),self.kind(op.results[0].type)),)
        elif name=='memref.dim':
            axis=int(operands[1])
            if not 0<=axis<len(operands[0].shape): raise SemanticFault('memref.dim 轴越界',rule_id='MEM-CONTRACT',line=line or None)
            values=(np.int64(operands[0].shape[axis]),)
        elif name=='memref.subview':
            base=operands[0];offsets,sizes,strides=self.decode_view_parameters(op,operands)
            spec=self.memtype(op.results[0].type);dims=spec.shape;choices=[]
            if len(dims)>len(sizes): raise Unsupported('subview 目标 rank 大于源 rank')
            for drop in itertools.combinations([i for i,n in enumerate(sizes) if n==1],len(sizes)-len(dims)):
                kept=[n for i,n in enumerate(sizes) if i not in drop]
                if all(d<0 or int(d)==n for d,n in zip(dims,kept)): choices.append(drop)
            if len(choices)!=1: raise Unsupported('subview 降维映射不唯一')
            buf=base.subview(offsets,sizes,strides,choices[0])
            self.view_layout(spec,buf)
            values=(buf,)
        elif name=='memref.reinterpret_cast':
            base=operands[0]
            self.view_layout(self.memtype(op.operands[0].type),base)
            offsets,sizes,strides=self.decode_view_parameters(op,operands)
            if len(offsets)!=1: raise Unsupported('reinterpret_cast 需要一个根存储 offset')
            spec=self.memtype(op.results[0].type)
            buf=base.reinterpret_cast(offsets[0],sizes,strides,element_type=self.kind(spec.element_type),line=line)
            self.view_layout(spec,buf)
            values=(buf,)
        elif name=='arith.constant':
            attr=op.attributes['value'];kind=self.kind(op.results[0].type)
            if isinstance(attr,(self.ir.IntegerAttr,self.ir.BoolAttr)): value=attr.value
            elif isinstance(attr,self.ir.FloatAttr):
                value=attr.value
                if np.isnan(value): raise Unsupported('FloatAttr 的 NaN payload 尚未核实；不通过宿主 float 近似')
            else: raise Unsupported('仅支持标量 IntegerAttr/FloatAttr 常量')
            values=(cast(value,kind),)
        elif name in MLIR_BATCH_SCALARS:
            kind=self.kind(op.results[0].type)
            types=[self.kind(arg.type) for arg in op.operands]
            values=(scalar_op(MLIR_BATCH_SCALARS[name][0],operands,kind,types),)
        elif name in self.binary:
            kind=self.kind(op.results[0].type)
            values=(apply(self.binary[name],operands,kind),)
        elif name in self.unary: values=(apply(self.unary[name],operands,self.kind(op.results[0].type)),)
        elif name in self.casts:
            source,target=self.kind(op.operands[0].type),self.kind(op.results[0].type)
            values=(mlir_cast(name.split('.')[1],operands[0],source,target),)
        elif name=='arith.cmpi':
            predicate=int(op.attributes['predicate'].value)
            predicates=('eq','ne','slt','sle','sgt','sge','ult','ule','ugt','uge')
            values=(mlir_cmpi(predicates[predicate],*operands,self.kind(op.operands[0].type)),)
        elif name=='arith.cmpf':
            pred = MLIR_CMPF_PREDICATES[int(op.attributes['predicate'].value)]
            values=(mlir_cmpf(pred,*operands),)
        elif name=='arith.select': values=(select(*operands),)
        else: raise Unsupported(f'未支持官方 MLIR 操作 {name}')
        if len(values)!=len(op.results): raise Unsupported(f'{name} 的结果数量不匹配')
        for key,value in zip(op.results,values): env[key]=trace_value(value,self.paths[op],line,name)
        return None

    def run(self,buffers,symbols=None,symbolic=False):
        self.budget=buffers[0].budget if buffers else Budget(self.max_steps)
        self.symbolic,self.pure_only,self.writer_locations,self.reductions=symbolic,False,{},[]
        with self.context:
            block=self.function.regions[0].blocks[0]
            if len(block.arguments)!=len(buffers): raise Unsupported('MLIR 参数数量与用例不同')
            env={}
            for arg,buf in zip(block.arguments,buffers):
                spec=self.memtype(arg.type)
                if len(spec.shape)!=len(buf.shape) or any(d>=0 and d!=n for d,n in zip(spec.shape,buf.shape)):
                    raise Unsupported('MLIR 参数 shape 与用例不同')
                if storage_type(dtype(self.kind(spec.element_type)))!=storage_type(buf.dtype):
                    raise Unsupported('MLIR 参数 dtype 与用例不同')
                self.layout(spec,buf);env[arg]=buf
            returned=self.sequence(block,env)
            if returned is None or returned[0]!='func.return' or returned[1]:
                raise Unsupported('目前只支持通过 memref 输出的 void 函数')
        return {'steps':self.budget.steps,'backend':'mlir_operation_objects',
                'execution_model':'structured logical loops; serial enumeration of parallel iterations',
                'location_model':'official parsed locations when checked; otherwise operation paths',
                'reductions':self.reductions}
