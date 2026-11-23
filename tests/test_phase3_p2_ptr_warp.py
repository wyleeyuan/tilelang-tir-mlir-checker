"""Numerical and refusal contracts for RISC-V pointer/shuffle replay."""


import numpy as np
import pytest

from tir_mlir_checker.execution import (
    Budget, Buffer, Expr, SemanticFault, Unsupported, apply, tir_scalar,
)
from tir_mlir_checker.tir_riscv_contracts import PointerView, shuffle_source, stream_words
from tests.test_phase3_p1_tir import block, func, launch, round_trip, run, tvm  # noqa: F401


@pytest.mark.parametrize('kind', ['float32', 'float64'])
def test_ieee_reciprocal_matches_same_type_division(kind):
    x = np.array([4, -2, 0., -0., np.inf, -np.inf, np.nan],dtype=kind)
    with np.errstate(all='ignore'):
        expected = np.divide(np.array(1,dtype=kind),x)
    result = np.array([tir_scalar('tl.ieee_frcp',[v,'rn'],kind,[kind,'str']) for v in x])
    np.testing.assert_array_equal(result,expected)
    np.testing.assert_array_equal(np.signbit(result[:-1]),np.signbit(expected[:-1]))
    symbol = Expr('input',(0,()),{'float32':'f32','float64':'f64'}[kind],scalar_dtype=kind)
    value = tir_scalar('tl.ieee_frcp',[symbol,'rn'],kind,[kind,'str'])
    assert value.fingerprint()==apply('div',[np.dtype(kind).type(1),symbol],kind).fingerprint()


@pytest.mark.parametrize('args,target', [((np.float32(1),'rz'),'float32'),
                                       ((np.float32(1),'rn'),'float64'),
                                       ((np.float16(1),'rn'),'float16')])
def test_reciprocal_refuses_other_modes_and_types(args,target):
    with pytest.raises(Unsupported):
        tir_scalar('tl.ieee_frcp',args,target)


def test_bounded_reference_aliases_root_and_tracks_conflicts():
    buf = Buffer(np.zeros((2,4),np.int32),initialized=False,symbolic=False,budget=Budget())
    ptr = PointerView(buf,3,3,3)
    assert ptr.index(1,'write')==(1,0)
    with pytest.raises(SemanticFault,match='未初始化'):
        ptr.read(1)
    region = buf.budget.new_region()
    with buf.budget.iteration(region,0):
        buf.write(ptr.index(1,'write'),7,0)
    assert buf.initialized[1,0] and buf.writes[1,0]==1
    assert ptr.read(1)==7
    with buf.budget.iteration(region,1), pytest.raises(SemanticFault) as fault:
        ptr.read(1)
    assert fault.value.rule_id=='PARALLEL-CONFLICT'
    with pytest.raises(SemanticFault) as fault:
        PointerView(buf,7,2,2)
    assert fault.value.rule_id=='MEM-OUT-OF-BOUNDS'
    with pytest.raises(Unsupported,match='rw_mask'):
        PointerView(buf,0,1,1).index(0,'write')


def test_stream_copy_16_bytes_mixed_storage_and_view():
    raw = np.array([0x3f800000,0x80000000,0x7fc01102,0x1234abcd],np.uint32)
    buf = Buffer(raw,initialized=True,symbolic=False)
    dest = Buffer(np.zeros(8,np.uint16),initialized=False,symbolic=False)
    copied = stream_words(PointerView(buf,0,4,1),PointerView(dest,0,8,2))
    np.testing.assert_array_equal(copied,raw.view(np.uint16))
    lanes = Buffer(np.arange(16,dtype=np.uint8),initialized=True,symbolic=False)
    np.testing.assert_array_equal(stream_words(PointerView(lanes,0,16,1),PointerView(buf,0,4,2)),
                                  lanes.values.view(np.uint32))


def test_shuffle_source_checks_invalid_backend_mask_shift():
    assert shuffle_source('down',31,32,1,32,0xffffffff,64)==31
    with pytest.raises(SemanticFault,match='超出位宽') as fault:
        shuffle_source('down',31,32,1,32,0xffffffff)
    assert fault.value.rule_id=='ARITH-INVALID-SHIFT'


def call(tvm,dtype,op,*args):
    return tvm.tir.Call(dtype,tvm.ir.Op.get(op),list(args))


def pointer(tvm, buf, offset, extent, mask, element=None):
    ptype = call(tvm,element or str(buf.dtype),'tir.type_annotation')
    return call(tvm,'handle','tir.tvm_access_ptr',ptype,buf.data,offset,extent,mask)


def test_real_reciprocal_mode_is_preflight_checked(tvm):
    t=tvm.tir
    a,b=[t.decl_buffer((1,),'float32',name=n) for n in 'ab']
    expr=call(tvm,'float32','tl.ieee_frcp',a[0],t.StringImm('rn'))
    np.testing.assert_array_equal(run(round_trip(tvm,func(t,t.BufferStore(b,expr,[0]),a,b)),
                                      np.array([-4],np.float32),np.empty(1,np.float32))[0].values,[-.25])
    bad=call(tvm,'float32','tl.ieee_frcp',a[0],t.StringImm('ru'))
    with pytest.raises(Unsupported,match='ieee_frcp'):
        round_trip(tvm,func(t,t.IfThenElse(False,t.BufferStore(b,bad,[0]),None),a,b))


def test_real_streaming_address_copy_gathers_before_aliasing_write(tvm):
    t=tvm.tir
    a=t.decl_buffer((6,),'int32',name='a')
    buf=t.decl_buffer((6,),'int32',name='tmp',scope='local')
    dst=t.decl_buffer((4,),'int32',name='out')
    address=lambda b,i: call(tvm,'handle','tir.address_of',b[i])
    stream=lambda b,i,j: t.Evaluate(t.call_extern('handle','tileops::load16_evict_last',address(b,i),address(b,j)))
    program=t.SeqStmt([*(t.BufferStore(buf,a[i],[i]) for i in range(4)),stream(buf,1,0),
                       *(t.BufferStore(dst,buf[i+1],[i]) for i in range(4))])
    f=func(t,block(t,program,[buf]),a,dst)
    out,info=run(round_trip(tvm,f),np.array([3,7,5,8,0,0],np.int32),np.empty(4,np.int32))
    np.testing.assert_array_equal(out.values,[3,7,5,8])
    assert any(e.get('action')=='stream_copy_16_bytes' for e in info['node_events'])
    bad=func(t,block(t,stream(buf,4,0),[buf]),a,dst)
    with pytest.raises(SemanticFault) as fault:
        run(round_trip(tvm,bad),np.ones(6,np.int32),np.empty(4,np.int32))
    assert fault.value.rule_id=='MEM-OUT-OF-BOUNDS'


def test_real_access_ptr_bounded_serial_load_and_store(tvm):
    t=tvm.tir
    a=t.decl_buffer((2,2),'int32',name='a')
    out=t.decl_buffer((2,2),'int32',name='out')
    get=lambda n: call(tvm,'int32','tl.atomic_load_elem_op',pointer(tvm,a,n,1,1),0)
    put=lambda n,mask: t.Evaluate(call(tvm,'handle','tl.atomic_store_elem_op',pointer(tvm,out,n,1,mask),get(3),0))
    f=func(t,t.SeqStmt([put(2,2),t.BufferStore(out,out[1,0],[0,0])]),a,out)
    res,_=run(round_trip(tvm,f),np.array([[4,6],[1,9]],np.int32),np.empty((2,2),np.int32))
    assert res.values[1,0]==res.values[0,0]==9
    assert res.initialized.tolist()==[[True,False],[True,False]]
    with pytest.raises(Unsupported,match='rw_mask'):
        round_trip(tvm,func(t,put(2,1),a,out))
    with pytest.raises(SemanticFault) as fault:
        run(round_trip(tvm,func(t,put(4,2),a,out)),np.zeros((2,2),np.int32),np.empty((2,2),np.int32))
    assert fault.value.rule_id=='MEM-OUT-OF-BOUNDS'
    with pytest.raises(Unsupported,match='未建模的消费点'):
        round_trip(tvm,func(t,t.Evaluate(pointer(tvm,a,0,1,1)),a,out))
    tx=t.Var('tx','int32')
    with pytest.raises(Unsupported,match='无并发'):
        round_trip(tvm,func(t,launch(t,tx,'threadIdx.x',32,put(2,2)),a,out))


def warp_program(tvm, op, mode, *, extent=32, width=32, guarded=False, mask=0xffffffff,
                 sync=False):
    t=tvm.tir
    tx,k=t.Var('tx','int32'),t.Var('k','int32')
    a,b=[t.decl_buffer((extent,),'int32',name=n) for n in ('a','b')]
    tmp=t.decl_buffer((1,),'int32',name='private',scope='local')
    amount=call(tvm,'int32','tir.shift_left',1,k) if mode=='sum' else (tx+1 if mode=='idx' else 1)
    extra=[32] if op.startswith('tir.') else []
    shuffle=call(tvm,'int32',op,t.IntImm('uint64',mask),tmp[0],amount,width,*extra)
    if mode=='sum':
        st=t.BufferStore(tmp,tmp[0]+shuffle,[0])
        st=t.For(k,0,5,t.ForKind.UNROLLED,st,annotations={'pragma_unroll_explicit':False})
        tail=[st,t.BufferStore(b,tmp[0],[tx])]
    else:
        stmt=t.BufferStore(b,shuffle,[tx])
        tail=[t.IfThenElse(tx<16,stmt,None) if guarded else stmt]
    if sync:
        tail.insert(0,t.Evaluate(call(tvm,'int32','tir.tvm_storage_sync',t.StringImm('shared'))))
    body=t.SeqStmt([t.BufferStore(tmp,a[tx],[0]),*tail])
    return func(t,launch(t,tx,'threadIdx.x',extent,block(t,body,[tmp])),a,b)


@pytest.mark.parametrize('op,mode',[('tl.shfl_xor_sync','sum'),('tl.shfl_sync','idx'),
                                    ('tir.tvm_warp_shuffle','idx'),('tir.tvm_warp_shuffle_down','down'),
                                    ('tl.shfl_up_sync','up')])
def test_real_shuffle_replays_per_lane_in_two_warps(tvm,op,mode):
    f=warp_program(tvm,op,mode,extent=64,width=32 if mode=='sum' else 8,sync=True)
    x=np.arange(64,dtype=np.int32)
    result,info=run(round_trip(tvm,f),x,np.empty_like(x),steps=30000)
    assert 'lane exchanges' in info['execution_model']
    if mode=='sum':
        expected=np.repeat(x.reshape(2,32).sum(axis=1),32)
    elif mode=='idx':
        expected=x//8*8+(x+1)%8
    elif mode=='down':
        expected=np.where(x%8==7,x,x+1)
    else:
        expected=np.where(x%8==0,x,x-1)
    np.testing.assert_array_equal(result.values,expected)


@pytest.mark.parametrize('kwargs,reason',[({'extent':16},'完整'),({'guarded':True},'一致到达'),
                                         ({'width':7},'width')])
def test_real_shuffle_rejects_unproven_replay(tvm,kwargs,reason):
    with pytest.raises(Unsupported,match=reason):
        round_trip(tvm,warp_program(tvm,'tl.shfl_xor_sync','xor',**kwargs))


def test_real_warp_mask_limits_source_and_does_not_clear_shared_conflict(tvm):
    t=tvm.tir
    x=np.arange(32,dtype=np.int32)
    f=warp_program(tvm,'tl.shfl_sync','idx',width=8,mask=0x55555555)
    out,_=run(round_trip(tvm,f),x,np.empty_like(x),steps=10000)
    source=x//8*8+(x+1)%8
    np.testing.assert_array_equal(out.values,np.where(source%2==0,source,x))

    tx=t.Var('tx','int32')
    a,b=[t.decl_buffer((32,),'int32',name=n) for n in ('a','b')]
    scratch=t.decl_buffer((1,),'int32',scope='shared')
    priv=t.decl_buffer((1,),'int32',scope='local')
    shuffled=call(tvm,'int32','tl.shfl_xor_sync',t.IntImm('uint32',0xffffffff),priv[0],1,32)
    program=t.SeqStmt([t.BufferStore(priv,a[tx],[0]),
        t.IfThenElse(t.EQ(tx,0),t.BufferStore(scratch,42,[0]),None),
        t.BufferStore(priv,shuffled,[0]),
        t.IfThenElse(t.EQ(tx,1),t.BufferStore(b,scratch[0],[tx]),None)])
    f=func(t,launch(t,tx,'threadIdx.x',32,block(t,program,[scratch,priv])),a,b)
    with pytest.raises(SemanticFault) as fault:
        run(round_trip(tvm,f),x,np.empty_like(x),steps=10000)
    assert fault.value.rule_id=='PARALLEL-CONFLICT'
