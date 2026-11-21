"""Shared operation names; detection is not a full syntax/type verifier."""
# Canonical scalar names shared by the execution frontends.
TIR_BATCH_SCALARS = {
    **{'tir.' + name: name for name in ('isnan', 'isfinite', 'log2', 'erf', 'copysign',
        'fmod', 'floor', 'trunc', 'sin', 'cos', 'ceil', 'log1p')},
    'tl.infinity': 'infinity', 'tir.infinity': 'infinity',
    'tir.round': 'roundaway', 'tir.nearbyint': 'roundeven', 'tir.pow': 'powf',
    'tl.ieee_fdiv': 'ieee_div', 'tl.ieee_fmaf': 'fma',
    'tir.bitwise_and': 'bitand', 'tir.bitwise_or': 'bitor', 'tir.bitwise_xor': 'bitxor',
    'tir.bitwise_not': 'bitnot', 'tir.shift_left': 'shl', 'tir.shift_right': 'ashr',
    'tir.popcount': 'ctpop', 'tir.clz': 'ctlz', 'tir.reinterpret': 'bitcast',
}
TIR_BATCH_SCRIPT = {'T.' + op.split('.')[1]: op for op in TIR_BATCH_SCALARS}
TIR_BATCH_SCRIPT.update({'T.power': 'tir.pow', 'T.infinity': 'tl.infinity'})
# Python spelling used by the TVMScript printer for some TIR calls.
TIR_BATCH_INFIX = {'BitAnd': 'tir.bitwise_and', 'BitOr': 'tir.bitwise_or',
                   'BitXor': 'tir.bitwise_xor', 'LShift': 'tir.shift_left',
                   'RShift': 'tir.shift_right'}

TIR_BATCH2_SCRIPT = {'T.'+name: 'tl.tileop.'+name for name in
                     ('fill','async_copy','tma_copy','gemm','gemm_py','wgmma_gemm')}
TIR_BATCH2_TILES = frozenset(TIR_BATCH2_SCRIPT.values())
TIR_BREAKS = frozenset({'tl.loop_break','tl.break_loop','tir.break_loop'})
TIR_SERIAL_MARKERS = frozenset({
    'tir.ptx_wait_group','tir.ptx_commit_group','tl.wait_wgmma','tl.set_max_nreg',
    'tl.no_set_max_nreg','tl.warpgroup_fence_operand','tl.fence_proxy_async','tl.ptx_fence_barrier_init',
})
TIR_REDUCE_COMBINE = {'sum':'add', 'abssum':'add', 'max':'max', 'absmax':'max', 'min':'min',
                      'bitand':'bitand', 'bitor':'bitor', 'bitxor':'bitxor'}
TIR_CUMSUM = 'tl.tileop.cumsum'
TIR_TUPLE = 'tir.tvm_tuple'
TIR_EXTERN = 'tir.call_extern'
TIR_EXTERN_SIGNATURES = {
    '__fdividef': ('float32',('float32','float32')),
    '__float2half_rz': ('float16',('float32',)),
}
TIR_TAIL_STATEMENTS = TIR_BREAKS | TIR_SERIAL_MARKERS | {TIR_CUMSUM,TIR_TUPLE}
TIR_TAIL_SCRIPT = {'T.'+op.rsplit('.',1)[1]: op for op in TIR_TAIL_STATEMENTS | {TIR_EXTERN}}
TIR_TAIL_SCRIPT['T.break_loop'] = 'tir.break_loop'
TIR_INTRINSICS = set(TIR_BATCH_SCALARS) | TIR_BATCH2_TILES | TIR_TAIL_STATEMENTS | {TIR_EXTERN,'tl.tileop.region'}

MLIR_BATCH_SCALARS = {
    **{'math.' + name: (name, 1) for name in
        ('log2', 'erf', 'floor', 'sin', 'cos', 'roundeven', 'trunc', 'ctpop', 'ctlz', 'log1p', 'ceil')},
    'math.copysign': ('copysign', 2), 'math.powf': ('powf', 2), 'math.fma': ('fma', 3),
    'arith.remf': ('fmod', 2),
    'arith.andi': ('bitand', 2), 'arith.ori': ('bitor', 2), 'arith.xori': ('bitxor', 2),
    'arith.shli': ('shl', 2), 'arith.shrsi': ('ashr', 2), 'arith.shrui': ('lshr', 2),
    'arith.divsi': ('sdiv', 2), 'arith.divui': ('udiv', 2), 'arith.remui': ('urem', 2),
    'arith.minsi': ('smin', 2), 'arith.maxsi': ('smax', 2),
    'arith.minui': ('umin', 2), 'arith.maxui': ('umax', 2),
}
MLIR_FLOAT_TO_INT = frozenset({'arith.fptosi','arith.fptoui'})
MLIR_BATCH_CASTS = {'arith.bitcast'} | MLIR_FLOAT_TO_INT
MLIR_MATMUL = frozenset({'linalg.matmul','linalg.matmul_transpose_a','linalg.matmul_transpose_b'})
MLIR_BATCH2_OPS = MLIR_MATMUL | {'linalg.yield','memref.reinterpret_cast'}
AFFINE_MATMUL_MAP = r'affine_map<\s*\(\s*d0\s*,\s*d1\s*,\s*d2\s*\)\s*->\s*\(\s*(d[012])\s*,\s*(d[012])\s*\)\s*>'
MAP_ALIAS = r'#[\w.$-]+'

TIR_METADATA = {"T.func_attr", "T.reads", "T.writes", "T.attr", "T.target", "T.block", "T.block_attr"}
TIR_CALLS = TIR_METADATA | set(TIR_BATCH_SCRIPT) | set(TIR_BATCH2_SCRIPT) | set(TIR_TAIL_SCRIPT) | {'T.evaluate','T.call_intrin','T.Ramp'} | {"T.match_buffer", "T.alloc_buffer", "T.launch_thread", "T.region",
    "T.copy", "T.fill", "T.min", "T.max", "T.exp", "T.exp2", "T.rsqrt", "T.sqrt", "T.sigmoid", "T.handle", "T.tvm_storage_sync", "T.if_then_else", "T.Cast", "T.assume", "T.reduce",
    "T.serial", "T.parallel", "T.unroll", "T.vectorized", "range"}
TIR_SCALARS = {"T." + n for n in ["int8", "uint8", "int16", "uint16", "int32", "uint32",
    "int64", "uint64", "float32", "float64", "float16", "bfloat16", "float8_e4m3fn", "bool"]}
MLIR_OPS = set(MLIR_BATCH_SCALARS) | MLIR_BATCH_CASTS | MLIR_BATCH2_OPS | {"memref.alloca", "memref.alloc", "memref.load", "memref.store", "memref.dim",
    "arith.constant", "arith.index_cast", "arith.index_castui", "arith.extsi", "arith.extui", "arith.trunci", "arith.uitofp", "arith.addi", "arith.subi", "arith.muli",
    "arith.addf", "arith.subf", "arith.mulf", "arith.divf",
    "memref.copy", "arith.negf", "arith.cmpf", "arith.floordivsi", "arith.remsi", "arith.sitofp", "math.exp", "math.exp2", "math.absf", "math.rsqrt", "math.sqrt", "memref.subview", "arith.cmpi", "arith.extf", "arith.truncf", "arith.select", "linalg.fill",
    "scf.for", "scf.parallel", "scf.if", "scf.while", "scf.condition", "scf.yield", "scf.reduce", "return"}
LOW_PRECISION = {"bf16", "bfloat16", "f16", "float16", "f8E4M3FN", "float8_e4m3fn"}
