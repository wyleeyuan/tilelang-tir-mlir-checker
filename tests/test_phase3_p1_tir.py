"""RISC-V local initialization, shared barriers, and checked layout metadata."""

import numpy as np
import pytest

from tir_mlir_checker.execution import Budget, SemanticFault, Unsupported, make_buffers
from tir_mlir_checker.tvm_interpreter import TVMInterpreter


@pytest.fixture(scope='module')
def tvm():
    return pytest.importorskip('tilelang').tvm


def run(engine, *arrays, symbolic=False, steps=2000):
    buffers = make_buffers(arrays, (len(arrays)-1,), symbolic, Budget(steps))
    info = engine.run(buffers, symbolic=symbolic)
    return buffers[-1], info


def block(t, body, buffers=(), annotations=None):
    return t.Block([], [], [], 'scope', body, alloc_buffers=list(buffers), annotations=annotations)


def func(t, body, *params):
    return t.PrimFunc([b.data for b in params], body, buffer_map={b.data:b for b in params})


def round_trip(tvm, f):
    return TVMInterpreter(tvm.ir.load_json(tvm.ir.save_json(f)))


def barrier(tvm, scope='shared', more=()):
    t = tvm.tir
    scope = t.StringImm(scope) if isinstance(scope,str) else scope
    return t.Evaluate(t.Call('int32', tvm.ir.Op.get('tir.tvm_storage_sync'), [scope,*more]))


def launch(t, var, tag, extent, body):
    axis = t.IterVar(None,var,t.IterVar.ThreadIndex,tag)
    return t.AttrStmt(axis,'thread_extent',extent,body)


def test_local_init_updates_only_zero_index_and_allows_later_write(tvm):
    t = tvm.tir
    local = t.decl_buffer((2,2),'int32',name='local',scope='local.var')
    out = t.decl_buffer((3,),'int32',name='out')
    init = 'tl.local_var_init'
    body = t.SeqStmt([t.BufferStore(out,local[0,0],[0]), t.BufferStore(local,9,[0,0]),
                      t.BufferStore(out,local[0,0],[1]), t.BufferStore(local,6,[1,1]),
                      t.BufferStore(out,local[1,1],[2])])
    # Block annotation wins over PrimFunc defaults, and the referenced Var is
    # the live scalar bound outside the block, not the name of a buffer.
    seed = t.Var('seed','int32')
    b = block(t,body,[local],{init:{local.data:seed+3}})
    f = func(t,t.LetStmt(seed,4,b),out).with_attr(init,{local.data:t.IntImm('int32',2)})
    engine = round_trip(tvm,f)
    output,info = run(engine,np.empty(3,np.int32))
    np.testing.assert_array_equal(output.values,[7,9,6])
    assert sum(e.get('action')=='local_var_init' for e in info['node_events'])==1
    out_symbol,_ = run(engine,np.empty(3,np.int32),symbolic=True)
    assert all(v.fingerprint() for v in out_symbol.expressions)

    # PrimFunc-only initialization and a compatible integer-to-float cast.
    fp = t.decl_buffer((1,),'float32',scope='local')
    fpout = t.decl_buffer((1,),'float32')
    plain = func(t,block(t,t.BufferStore(fpout,fp[0],[0]),[fp]),fpout)
    defaults = plain.with_attr(init,{fp.data:t.IntImm('int32',3)})
    np.testing.assert_array_equal(run(round_trip(tvm,defaults),np.empty(1,np.float32))[0].values,[3.0])
    with pytest.raises(SemanticFault) as problem:
        run(round_trip(tvm,plain),np.empty(1,np.float32))
    assert problem.value.rule_id=='MEM-UNINITIALIZED'

    unwritten = func(t,block(t,t.BufferStore(out,local[1,1],[0]),[local],
                             {init:{local.data:t.IntImm('int32',7)}}),out)
    with pytest.raises(SemanticFault) as problem:
        run(round_trip(tvm,unwritten),np.empty(3,np.int32))
    assert problem.value.rule_id=='MEM-UNINITIALIZED'


def test_local_init_preflight_rejects_bad_mapping_and_values(tvm):
    t = tvm.tir
    out = t.decl_buffer((1,),'int32')
    local = t.decl_buffer((1,),'int32',scope='local.var')
    bad_maps = [t.IntImm('int32',1), {out.data:1}, {local:1}, {local.data:t.StringImm('bad')},
                {local.data:t.Var('unbound','int32')}, {local.data:out[0]}, {local.data:1.25}]
    for bad in bad_maps:
        with pytest.raises(Unsupported,match='tl.local_var_init'):
            TVMInterpreter(func(t,block(t,t.Evaluate(0),[local],{'tl.local_var_init':bad}),out))
    with pytest.raises(Unsupported,match='tl.local_var_init'):
        TVMInterpreter(func(t,t.Evaluate(0),out).with_attr('tl.local_var_init',{local.data:1}))


def test_serial_and_unit_thread_barriers(tvm):
    t = tvm.tir
    out = t.decl_buffer((1,),'int32')
    var = t.Var('tid','int32')
    body = t.SeqStmt([barrier(tvm),t.BufferStore(out,7,[0]),barrier(tvm,'shared.dyn')])
    for program in (body, launch(t,var,'threadIdx.x',1,body)):
        output,info = run(round_trip(tvm,func(t,program,out)),np.empty(1,np.int32))
        assert output.values.item()==7
        assert [e['scope'] for e in info['node_events'] if e['action']=='serial_barrier']==['shared','shared.dyn']
        assert info['steps'] >= 3


def phase_program(tvm, *, collision=None, repeat=False, blocks=1):
    t = tvm.tir
    src = t.decl_buffer((blocks*2,),'int32',name='src')
    out = t.decl_buffer((blocks*2,),'int32',name='out')
    shared = t.decl_buffer((2,),'int32',name='shared_lab',scope='shared.dyn')
    local = t.decl_buffer((1,),'int32',name='private',scope='local.var')
    tx = t.Var('tx','int32')
    bx = t.Var('bx','int32')
    input_pos = bx*2+tx if blocks>1 else tx
    output_pos = input_pos if collision!='interblock' else tx
    body = t.SeqStmt([t.BufferStore(shared,src[input_pos]+local[0],[0 if collision=='write' else tx]),
                      barrier(tvm,'shared.dyn'),
                      t.BufferStore(out,shared[1-tx],[output_pos]),
                      t.BufferStore(shared,5,[tx]) if collision=='readwrite' else barrier(tvm)])
    if repeat:
        body = t.For(t.Var('i','int32'),0,2,t.ForKind.SERIAL,body)
    regions = block(t,body,[shared,local],{'tl.local_var_init':{local.data:tx+10}})
    body = launch(t,tx,'threadIdx.x',2,regions)
    if blocks>1:
        body = launch(t,bx,'blockIdx.x',blocks,body)
    return func(t,body,src,out)


def test_real_tvm_barrier_visibility_private_state_and_static_loop(tvm):
    for repeat in (False,True):
        engine = round_trip(tvm,phase_program(tvm,repeat=repeat,blocks=2))
        output,info = run(engine,np.asarray([1,8,3,20],np.int32),np.empty(4,np.int32))
        np.testing.assert_array_equal(output.values,[19,11,31,13])
        assert all(output.initialized)
        assert sum(e.get('action')=='thread_phase_barrier' for e in info['node_events'])==(16 if repeat else 8)
        assert np.all(output.writes==(2 if repeat else 1))
        symbol,_ = run(engine,np.arange(4,dtype=np.int32),np.empty(4,np.int32),symbolic=True)
        assert all(v.fingerprint() for v in symbol.expressions)
    with pytest.raises(Unsupported,match='步数上限'):
        run(engine,np.arange(4,dtype=np.int32),np.empty(4,np.int32),steps=6)


@pytest.mark.parametrize('collision,blocks',[('write',1),('readwrite',1),('interblock',2)])
def test_barrier_does_not_hide_conflicts_within_phase_or_between_blocks(tvm,collision,blocks):
    engine = round_trip(tvm,phase_program(tvm,collision=collision,blocks=blocks))
    with pytest.raises(SemanticFault) as problem:
        run(engine,np.arange(blocks*2,dtype=np.int32),np.empty(blocks*2,np.int32))
    assert problem.value.rule_id=='PARALLEL-CONFLICT'


def test_barrier_preflight_rejects_conditional_nonuniform_and_unknown_scope(tvm):
    t = tvm.tir
    tid = t.Var('tid','int32')
    i = t.Var('i','int32')
    sync = barrier(tvm)
    for body,reason in [(t.IfThenElse(t.IntImm('bool',False),sync,None),'条件分支'),
                        (t.For(i,0,tid+1,t.ForKind.SERIAL,sync),'静态一致'),
                        (t.For(i,0,2,t.ForKind.PARALLEL,sync),'静态一致'),
                        (barrier(tvm,'global'),'作用域'),
                        (barrier(tvm,'shared',[t.IntImm('int32',2)]),'作用域'),
                        (barrier(tvm,tid),'作用域')]:
        with pytest.raises(Unsupported,match=reason):
            TVMInterpreter(func(t,launch(t,tid,'threadIdx.x',2,body)))
    with pytest.raises(Unsupported,match='单一 threadIdx.x'):
        TVMInterpreter(func(t,launch(t,tid,'threadIdx.y',2,sync)))
    var = t.IterVar(None,tid,t.IterVar.ThreadIndex,'threadIdx.x')
    bound = t.For(i,0,2,t.ForKind.THREAD_BINDING,sync,thread_binding=var)
    with pytest.raises(Unsupported,match='AttrStmt thread_extent'):
        TVMInterpreter(func(t,bound))
    out = t.decl_buffer((1,),'int32')
    with pytest.raises(Unsupported,match='不能用作值'):
        TVMInterpreter(func(t,t.BufferStore(out,sync.value,[0]),out))


def test_static_row_major_layouts_hold_same_logical_coordinates(tvm):
    from tilelang.layout import Layout
    t = tvm.tir
    out = t.decl_buffer((1,),'int32')
    for shape,scope,mapping in [((4,), 'shared', lambda i: i),
                               ((1,4), 'shared.dyn', lambda i,j: j),
                               ((2,2,3), 'shared.dyn', lambda i,j,k: i*12+j*4+k),
                               ((2,2,3), 'shared', lambda i,j,k: (i,j,k))]:
        local = t.decl_buffer(shape,'int32',scope=scope)
        end = tuple(n-1 for n in shape)
        start = (0,)*len(shape)
        value = t.BufferLoad(local,list(end))+t.BufferLoad(local,list(start))
        statements = t.SeqStmt([t.BufferStore(local,7,list(start)),t.BufferStore(local,23,list(end)),
                                 t.BufferStore(out,value,[0])])
        body = block(t,statements,[local],{'layout_map':{local.data:Layout(list(shape),mapping)}})
        assert run(round_trip(tvm,func(t,body,out)),np.empty(1,np.int32))[0].values.item()==30


def test_layout_preflight_rejects_overlap_stride_and_swizzle(tvm):
    from tilelang.layout import Layout
    t = tvm.tir
    local = t.decl_buffer((2,3),'int32',scope='shared.dyn')
    for mapping in (lambda i,j: i*2+j, lambda i,j: i*3+j*2,
                    lambda i,j: i*3-j, lambda i,j: i*3+(j+i)%3):
        body = block(t,t.Evaluate(0),[local],{'layout_map':{local.data:Layout([2,3],mapping)}})
        with pytest.raises(Unsupported,match='layout_map'):
            TVMInterpreter(func(t,body))
