"""SCF while: before/after transfer, isolation, and bounded execution."""
import importlib
from contextlib import nullcontext

import numpy as np
import pytest

from tir_mlir_checker.execution import Budget, Unsupported, make_buffers
from tir_mlir_checker.mlir_interpreter import MLIRInterpreter
from tir_mlir_checker.mlir_object_interpreter import MLIRObjectInterpreter


def program(body, arguments='%out: memref<2xi32>'):
    return f'''module {{
  func.func @main({arguments}) {{
    %z = arith.constant 0 : index
    %o = arith.constant 1 : index
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %two = arith.constant 2 : i32
{body}
    return
  }}
}}
'''


def carried(limit):
    return program(f'''    %limit = arith.constant {limit} : i32
    %r:2 = scf.while (%i = %zero, %sum = %one) : (i32, i32) -> (i32, i32) {{
      %test = arith.cmpi slt, %i, %limit : i32
      %forward = arith.addi %sum, %two : i32
      scf.condition(%test) %i, %forward : i32, i32
    }} do {{
    ^bb0(%j: i32, %total: i32):
      %next = arith.addi %j, %one : i32
      %acc = arith.addi %total, %j : i32
      scf.yield %next, %acc : i32, i32
    }}
    memref.store %r#0, %out[%z] : memref<2xi32>
    memref.store %r#1, %out[%o] : memref<2xi32>''')


DIFFERENT_EDGES = program('''    %r:2 = scf.while (%i = %zero) : (i32) -> (i32, i1) {
      %test = arith.cmpi slt, %i, %two : i32
      scf.condition(%test) %i, %test : i32, i1
    } do {
    ^bb0(%j: i32, %still: i1):
      %next = arith.addi %j, %one : i32
      scf.yield %next : i32
    }
    %flag = arith.extui %r#1 : i1 to i32
    memref.store %r#0, %out[%z] : memref<2xi32>
    memref.store %flag, %out[%o] : memref<2xi32>''')


SINGLE_RESULT = program('''    %r = scf.while (%i = %zero) : (i32) -> i32 {
      %next = arith.addi %i, %one : i32
      %keep = arith.cmpi slt, %next, %two : i32
      scf.condition(%keep) %next : i32
    } do {
    ^bb0(%j: i32):
      scf.yield %j : i32
    }
    memref.store %r, %out[%z] : memref<2xi32>
    memref.store %one, %out[%o] : memref<2xi32>''')


NO_ARGS = program('''    memref.store %zero, %out[%z] : memref<2xi32>
    %yes = arith.constant true
    scf.if %yes {
      scf.while : () -> () {
        %i = memref.load %out[%z] : memref<2xi32>
        %test = arith.cmpi slt, %i, %two : i32
        scf.condition(%test)
      } do {
        %i = memref.load %out[%z] : memref<2xi32>
        %next = arith.addi %i, %one : i32
        memref.store %next, %out[%z] : memref<2xi32>
        scf.yield
      }
    }
    memref.store %one, %out[%o] : memref<2xi32>''')


INFINITE = program('''    %yes = arith.constant true
    scf.while : () -> () {
      scf.condition(%yes)
    } do {
      scf.yield
    }''')


def object_engine(source, *, generic=False):
    from mlir import ir
    context = ir.Context()
    with context:
        for dialect in ('arith','func','memref','scf'):
            importlib.import_module('mlir.dialects.'+dialect)
        context.load_all_available_dialects()
        module = ir.Module.parse(source)
        assert module.operation.verify()
        if generic:
            source=module.operation.get_asm(print_generic_op_form=True)
            module=ir.Module.parse(source)
            assert module.operation.verify()
        return MLIRObjectInterpreter((context,module),source)


def run(engine, *, steps=200, symbolic=False):
    budget=Budget(steps)
    buffers=make_buffers((np.empty(2,np.int32),),(0,),symbolic,budget)
    info=engine.run(buffers,symbolic=symbolic)
    assert info['steps']==budget.steps
    return buffers[0]


def check_supported(factory):
    for source,expected in [(carried(0),(0,3)),(carried(3),(3,12)),
                            (DIFFERENT_EDGES,(2,0)),(SINGLE_RESULT,(2,1)),(NO_ARGS,(2,1))]:
        buf=run(factory(source))
        np.testing.assert_array_equal(buf.values,expected)
        assert buf.initialized.all()
    # With concrete control flow, carried symbolic values remain expressions.
    symbolic=run(factory(carried(3)),symbolic=True)
    assert all(v.fingerprint() for v in symbolic.expressions)
    with pytest.raises(Unsupported,match='步数上限'):
        run(factory(INFINITE),steps=20)
    with pytest.raises(Unsupported,match='符号 scf.while'):
        run(factory(NO_ARGS),symbolic=True)


def test_while_text():
    check_supported(MLIRInterpreter)


def test_while_real_mlir_objects():
    pytest.importorskip('mlir.ir')
    check_supported(object_engine)
    np.testing.assert_array_equal(run(object_engine(DIFFERENT_EDGES,generic=True)).values,[2,0])


@pytest.mark.parametrize('source,reason,at_run', [
    (INFINITE.replace('scf.condition(%yes)','scf.condition(%zero)'), '需要 i1', True),
    (INFINITE.replace('scf.condition(%yes)',''), '缺少 scf.condition', False),
    (INFINITE.replace('scf.yield',''), '缺少 scf.yield', False),
    (DIFFERENT_EDGES.replace('scf.yield %next : i32','scf.yield %next : i64'), '参数或类型不匹配', False),
    (INFINITE.replace('scf.yield','test.unmodeled\n      scf.yield'), '未支持 MLIR 操作 test.unmodeled', False),
    (INFINITE.replace('scf.yield','return\n      scf.yield'), '终结符位置不合法', False),
    (program('    scf.condition(%one)'), '终结符位置不合法', False),
    (program('    return\n  } do {\n    return').replace('@main','@named.scf.while'),
     '无法解析', False),
])
def test_while_text_rejects_invalid_or_unknown_ir(source,reason,at_run):
    with pytest.raises(Unsupported,match=reason):
        engine=MLIRInterpreter(source)
        if at_run:
            run(engine)


def test_while_real_mlir_rejects_invalid_terminators():
    ir=pytest.importorskip('mlir.ir')
    with ir.Context() as context:
        context.load_all_available_dialects()
        for source in (INFINITE.replace('scf.condition(%yes)','scf.condition(%zero)'),
                       INFINITE.replace('scf.condition(%yes)','scf.yield'),
                       INFINITE.replace('scf.yield','scf.condition(%yes)')):
            with pytest.raises(ir.MLIRError):
                module=ir.Module.parse(source)
                module.operation.verify()
