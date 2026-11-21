"""Shared bounded memory/scalar model; frontends do not translate to each other."""
from __future__ import annotations

import hashlib
import json
import math
from fractions import Fraction
from dataclasses import dataclass, field, replace
from contextlib import contextmanager

import numpy as np

from .capabilities import TIR_BATCH_SCALARS, TIR_EXTERN_SIGNATURES


class Unsupported(ValueError):
    """Outside the explicitly implemented IR semantics."""


class SemanticFault(ValueError):
    """Concrete violation of the IR execution contract for a supplied test case."""

    def __init__(self, message, *, rule_id='MEM-CONTRACT', line=None, **details):
        super().__init__(message)
        self.rule_id, self.line, self.details = rule_id, line, details


class LoopBreak(Exception):
    """Internal statement control signal, caught by the nearest serial For."""


def while_test(condition):
    """SCF control flow needs a concrete i1; never choose a symbolic path."""
    if isinstance(condition, Expr):
        raise Unsupported('数据相关符号 scf.while 条件；仅执行具体输入')
    if not isinstance(condition, (bool, np.bool_)):
        raise Unsupported('scf.condition 需要 i1')
    return bool(condition)


def dtype(name):
    aliases = {"f32": "float32", "f64": "float64", "i1": "bool", "index": "int64"}
    if name in ("f8E4M3FN", "float8_e4m3fn"):
        try:
            import ml_dtypes
        except ImportError as exc:
            raise Unsupported("E4M3FN 执行需要 ml_dtypes>=0.5；不使用 uint8 近似浮点数") from exc
        return np.dtype(ml_dtypes.float8_e4m3fn)
    if name in ("bf16", "bfloat16"):
        try:
            import ml_dtypes
        except ImportError as exc:
            raise Unsupported("bf16 执行需要可选依赖 ml_dtypes>=0.5；未安装时不近似模拟") from exc
        return np.dtype(ml_dtypes.bfloat16)
    if name in ("float16", "f16"):
        return np.dtype("float16")
    name=aliases.get(name,name)
    if name.startswith("i") and name[1:].isdigit():
        name = "int" + name[1:]
    try:
        result = np.dtype(aliases.get(name, name))
    except TypeError as exc:
        raise Unsupported(f"未支持的数据类型 {name}") from exc
    if result.kind not in "biuf" or result.itemsize > 8:
        raise Unsupported(f"未支持的数据类型 {name}")
    return result


def storage_type(value):
    d = np.dtype(value)
    if d.name == "bfloat16":
        return "bf16"
    if d.name == "float8_e4m3fn":
        return "f8E4M3FN"
    return ("f" if d.kind == "f" else "i") + str(d.itemsize * 8) if d.kind != "b" else "i1"


@dataclass(frozen=True, eq=False)
class Expr:
    op: str
    args: tuple
    type: str
    trace: tuple = field(default=(), compare=False, repr=False)
    # TIR needs unsignedness for dispatch; structural equality uses stored bits.
    scalar_dtype: str | None = field(default=None, compare=False, repr=False)

    def __bool__(self):
        raise Unsupported("符号条件不能作为宿主 Python 布尔值")

    def __eq__(self,other):
        if not isinstance(other,Expr):return False
        pending=[(self,other)];seen=set()
        while pending:
            a,b=pending.pop()
            if a is b:continue
            if isinstance(a,Expr) and isinstance(b,Expr):
                pair=(id(a),id(b))
                if pair in seen:continue
                seen.add(pair)
                if a.op!=b.op or a.type!=b.type or len(a.args)!=len(b.args):return False
                pending.extend(zip(a.args,b.args))
            elif isinstance(a,Expr) or isinstance(b,Expr) or a!=b:return False
        return True

    __hash__=None

    def fingerprint(self):
        pending=[(self,False)]
        while pending:
            value,ready=pending.pop()
            if hasattr(value,'_fingerprint'):continue
            if not ready:
                pending.append((value,True))
                pending.extend((a,False) for a in value.args if isinstance(a,Expr) and not hasattr(a,'_fingerprint'))
                continue
            args=[['expr',a._fingerprint] if isinstance(a,Expr) else ['literal',a] for a in value.args]
            raw=json.dumps([value.op,value.type,args],separators=(',',':'),default=lambda x: x.item() if hasattr(x,'item') else str(x)).encode()
            object.__setattr__(value,'_fingerprint',hashlib.sha256(raw).hexdigest())
        return self._fingerprint

    def data(self):
        # Diagnostics may display shared DAGs with exponential expanded size.
        # Equality above remains exact; only the human-facing tree is bounded.
        remaining=[2048]
        def show(value,depth):
            if not isinstance(value,Expr):return value.item() if isinstance(value,np.generic) else value
            remaining[0]-=1
            if depth>=32 or remaining[0]<=0:return ['truncated',value.type,value.fingerprint()]
            return [value.op,value.type,*[show(a,depth+1) for a in value.args]]
        return show(self,0)


def merge_trace(values):
    records = list(dict.fromkeys(t for v in values if isinstance(v, Expr) for t in v.trace))
    return tuple(records if len(records) <= 64 else records[:32] + records[-32:])


def provenance(value):
    return [{'buffer': b, 'line': line, 'action': action, 'index': list(index)}
            for b, line, action, index in value.trace] if isinstance(value, Expr) else []


def trace_value(value, symbol, line, action, index=()):
    if not isinstance(value, Expr):
        return value
    extra = Expr('location', (), value.type, ((symbol, int(line), action, tuple(index)),))
    return replace(value, trace=merge_trace((value, extra)))


def literal(value, type_name=None):
    if isinstance(value, Expr):
        return value
    value = np.asarray(value)
    if value.ndim:
        raise Unsupported("仅支持标量表达式")
    if type_name is not None:
        value = value.astype(dtype(type_name))
    return Expr("constant", (value.tobytes().hex(),), storage_type(value.dtype), scalar_dtype=value.dtype.name)


def apply(op, values, type_name=None):
    symbolic = any(isinstance(v, Expr) for v in values)
    if symbolic:
        nodes = tuple(literal(v) for v in values)
        if op in {"ne", "eq", "lt", "le", "gt", "ge", "and", "or", "not"}:
            # Equality is insensitive to integer signedness, but not width.
            result_type = "i1"
        else:
            result_type = storage_type(dtype(type_name)) if type_name else nodes[0].type
        declared = dtype(type_name).name if type_name else nodes[0].scalar_dtype
        return Expr(op, nodes, result_type, merge_trace(nodes), "bool" if result_type == "i1" else declared)
    operations = {"add": np.add, "sub": np.subtract, "mul": np.multiply,
                  "div": np.divide, "floordiv": np.floor_divide, "mod": np.remainder,
                  "ne": np.not_equal, "eq": np.equal, "lt": np.less, "le": np.less_equal,
                  "gt": np.greater, "ge": np.greater_equal,
                  "and": np.logical_and, "or": np.logical_or, "neg": np.negative,
                  "max": np.maximum, "min": np.minimum, "exp": np.exp, "exp2": np.exp2, "sqrt": np.sqrt, "log": np.log, "abs": np.abs, "tanh": np.tanh, "not": np.logical_not}
    if op == 'rsqrt':
        # MLIR math.rsqrt represents reciprocal square root. No algebraic
        # identity is used in symbolic comparisons.
        with np.errstate(all="ignore"):
            value = np.asarray(values[0])
            return cast(np.divide(np.asarray(1, dtype=value.dtype), np.sqrt(value)), type_name or str(value.dtype))
    if op == 'sigmoid':
        kind=type_name or str(np.asarray(values[0]).dtype)
        one=cast(1,kind)
        return apply('div', [one, apply('add', [one, apply('exp', [apply('neg',values)], kind)], kind)], kind)
    # Use exact Python integers before wrapping to the IR width. In particular,
    # int64 division/remainder must never pass through a host floating value.
    kind = dtype(type_name) if type_name else np.result_type(*values)
    if kind.kind in 'iu' and op in {'add','sub','mul','neg','floordiv','mod','rem'}:
        integers = list(map(int, values))
        a = integers[0]
        if op in {'floordiv','mod','rem'}:
            b = integers[1]
            if b == 0 or (kind.kind == 'i' and a == np.iinfo(kind).min and b == -1):
                raise SemanticFault('整数除法除零或有符号溢出', rule_id='ARITH-INVALID-DIVISION')
        if op == 'rem':
            q = (abs(a)//abs(b)) * (-1 if (a < 0) != (b < 0) else 1)
            answer = a-q*b
        else:
            answer = {'add':lambda:a+integers[1], 'sub':lambda:a-integers[1],
                      'mul':lambda:a*integers[1], 'neg':lambda:-a,
                      'floordiv':lambda:a//b, 'mod':lambda:a%b}[op]()
        return cast(answer, str(kind))
    if op in {'floordiv', 'mod', 'rem'} and not symbolic:
        a,b=values
        if int(b)==0 or (isinstance(a,np.signedinteger) and int(a)==np.iinfo(a.dtype).min and int(b)==-1):
            raise SemanticFault('整数除法除零或有符号溢出', rule_id='ARITH-INVALID-DIVISION')
    if op == 'rem':
        a,b=map(int,values)
        quotient=(abs(a)//abs(b)) * (-1 if (a<0)!=(b<0) else 1)
        return cast(a-quotient*b,type_name or str(np.asarray(values[0]).dtype))
    if op not in operations:
        raise Unsupported(f"未支持的标量运算 {op}")
    with np.errstate(all="ignore"):
        result = operations[op](*values)
    if type_name is not None:
        result = np.asarray(result, dtype=dtype(type_name))[()]
    return result


def cast(value, target):
    if isinstance(value, Expr):
        declared = dtype(target)
        kind = storage_type(declared)
        if value.type == kind:
            return value if value.scalar_dtype == declared.name else replace(value, scalar_dtype=declared.name)
        return Expr("cast", (value,), kind, value.trace, declared.name)
    kind = dtype(target)
    scalar = np.asarray(value)
    if scalar.ndim:
        raise Unsupported('类型转换仅接受标量')
    if kind.kind in 'iu' and (scalar.dtype.kind in 'biu' or type(value) is int):
        bits = kind.itemsize*8
        raw = int(value) % (1 << bits)
        if kind.kind == 'i' and raw >= 1 << (bits-1): raw -= 1 << bits
        return np.asarray(raw, dtype=kind)[()]
    with np.errstate(all='ignore'):
        return scalar.astype(kind)[()]


def integer_interpret(value, source, *, unsigned=False):
    """Interpret signless MLIR bits as signed/unsigned without changing width."""
    kind = dtype(source)
    if kind.kind not in 'biu': raise Unsupported('需要整数位类型')
    bits = 1 if kind.kind == 'b' else kind.itemsize*8
    if isinstance(value, Expr):
        # Keep the signedness choice in the expression: equal storage widths
        # alone do not make zero/sign extensions equivalent.
        return Expr('unsigned_bits' if unsigned else 'signed_bits', (value,),
                    ('uint' if unsigned else 'int')+str(max(8,bits)), value.trace)
    raw = int(value) % (1 << bits)
    return raw if unsigned or raw < 1 << (bits-1) else raw-(1 << bits)


def scalar_type(value):
    if isinstance(value, Expr):
        return value.scalar_dtype or value.type
    return np.asarray(value).dtype.name


def _float_type(kind):
    return kind.kind == 'f' or kind.name in {'bfloat16', 'float8_e4m3fn'}


def _width(kind):
    return 1 if kind.kind == 'b' else 8 * kind.itemsize


def _integer_result(value, kind):
    if kind.kind == 'b':
        return np.bool_(value & 1)
    return cast(value, kind.name)


def _from_bits(bits, kind):
    return np.asarray(bits, dtype=f'uint{kind.itemsize * 8}').view(kind)[()]


def float32_to_half_rz(value):
    """Truncate binary32 directly to binary16, including subnormals and signed zero."""
    if isinstance(value,Expr):
        return Expr('fptrunc_rz',(value,),'f16',value.trace,'float16')
    raw = int(np.asarray(value,dtype=np.float32).view(np.uint32)[()])
    sign, exp, frac = (raw >> 16) & 0x8000, (raw >> 23) & 0xff, raw & 0x7fffff
    power = exp-127
    if exp == 0xff:
        # Preserve sign and the leading payload bits; the conversion quiets NaNs.
        bits = (0x7e00 | (frac >> 13)) if frac else 0x7c00
    elif power > 15:
        bits = 0x7bff  # A finite overflow rounds toward the largest finite magnitude.
    elif power >= -14:
        bits = ((power+15) << 10) | (frac >> 13)
    elif power >= -24:
        bits = (0x800000 | frac) >> (-power-1)
    else:
        bits = 0
    return _from_bits(sign | bits,np.dtype('float16'))


def extern_scalar(symbol, values, target, sources):
    if TIR_EXTERN_SIGNATURES.get(symbol) != (target,tuple(sources)):
        raise Unsupported(f'tir.call_extern {symbol}: 不支持的符号、参数或 dtype')
    if len(values) != len(sources) or any(scalar_type(v) not in (kind,'f32') for v,kind in zip(values,sources)):
        raise Unsupported(f'tir.call_extern {symbol}: 实际参数或 dtype 不符')
    if symbol == '__float2half_rz':
        return float32_to_half_rz(values[0])
    lhs,rhs = values
    # Match LowerFastFloatDivide's f32 exceptional divisor substitution.
    infinity,limit = np.float32(np.inf),np.float32(2**126)
    if isinstance(rhs,Expr):
        mag = apply('abs',[rhs],'f32')
        def compare(predicate, other):
            args = (mag,literal(other))
            return Expr('cmpf_'+predicate,args,'i1',merge_trace(args))
        exceptional = apply('and',[compare('ogt',limit),compare('olt',infinity)],'bool')
        rhs = select(exceptional,scalar_op('copysign',[infinity,rhs],'f32'),rhs)
    elif limit < abs(rhs) < infinity:
        rhs = scalar_op('copysign',[infinity,rhs],'f32')
    return scalar_op('ieee_div',[lhs,rhs],'f32')


_FLOAT_FORMATS = {
    'float16': (5, 10), 'bfloat16': (8, 7), 'float32': (8, 23), 'float64': (11, 52),
}


def _rounded_float(value, kind, negative_zero=False):
    """Round an exact binary rational once to the destination's IEEE bits."""
    exponent_bits, fraction_bits = _FLOAT_FORMATS[kind.name]
    sign = int(value < 0 or (not value and negative_zero))
    sign_bits = sign << (exponent_bits + fraction_bits)
    if not value:
        return _from_bits(sign_bits, kind)
    numerator, denominator = abs(value.numerator), value.denominator
    exponent = numerator.bit_length() - denominator.bit_length()
    if (numerator < denominator << exponent if exponent >= 0
            else numerator << -exponent < denominator):
        exponent -= 1
    bias = (1 << (exponent_bits - 1)) - 1
    min_exponent = 1 - bias
    shift = fraction_bits - max(exponent, min_exponent)
    if shift >= 0:
        numerator <<= shift
    else:
        denominator <<= -shift
    significand, remainder = divmod(numerator, denominator)
    if 2 * remainder > denominator or (2 * remainder == denominator and significand & 1):
        significand += 1
    if exponent < min_exponent:
        # Also encodes a subnormal rounding up to the smallest normal.
        return _from_bits(sign_bits | significand, kind)
    if significand == 1 << (fraction_bits + 1):
        significand >>= 1
        exponent += 1
    encoded_exponent = exponent + bias
    if encoded_exponent >= (1 << exponent_bits) - 1:
        return _from_bits(sign_bits | (((1 << exponent_bits) - 1) << fraction_bits), kind)
    return _from_bits(sign_bits | (encoded_exponent << fraction_bits)
                      | (significand - (1 << fraction_bits)), kind)


def _fused_multiply_add(a, b, c, kind):
    # Widening these input formats to binary64 is exact. The finite product
    # and sum are rational, even on Python versions without math.fma. Never
    # round via a binary64 result on the way to f32, f16 or bf16.
    a, b, c = map(float, (a, b, c))
    if any(map(math.isnan, (a, b, c))):
        return cast(math.nan, kind.name)  # IEEE does not fix the NaN payload here.
    product_negative = (math.copysign(1, a) != math.copysign(1, b))
    if math.isinf(a) or math.isinf(b):
        if a == 0 or b == 0 or (math.isinf(c) and (c < 0) != product_negative):
            return cast(math.nan, kind.name)
        return cast(-math.inf if product_negative else math.inf, kind.name)
    if math.isinf(c):
        return cast(c, kind.name)
    exact = Fraction(a) * Fraction(b) + Fraction(c)
    negative_zero = (a == 0 or b == 0) and product_negative and c == 0 and math.copysign(1, c) < 0
    return _rounded_float(exact, kind, negative_zero)


def _round_away(value):
    value = float(value)
    if not math.isfinite(value):
        return value
    fraction, integral = math.modf(abs(value))
    # floor(abs(x) + 0.5) rounds the addition first and gets inputs just
    # below 0.5 wrong. Compare the exact fractional part instead.
    return math.copysign(integral + (fraction >= 0.5), value)


_FLOAT_UNARY = {
    'log2': np.log2, 'erf': math.erf, 'floor': np.floor, 'trunc': np.trunc,
    'sin': np.sin, 'cos': np.cos, 'roundaway': _round_away, 'roundeven': np.rint,
    'ceil': np.ceil, 'log1p': np.log1p, 'isnan': np.isnan, 'isfinite': np.isfinite,
}
_INTEGER_UNARY = {'bitnot', 'ctpop', 'ctlz'}
_INTEGER_BINARY = {'bitand', 'bitor', 'bitxor', 'shl', 'ashr', 'lshr',
                   'sdiv', 'udiv', 'urem', 'smin', 'smax', 'umin', 'umax', 'ipow', 'upow'}
_FLOAT_BINARY = {'fmod', 'copysign', 'powf', 'ieee_div'}
_SCALAR_ARITY = {**dict.fromkeys(set(_FLOAT_UNARY) | _INTEGER_UNARY | {'bitcast'}, 1),
                 **dict.fromkeys(_INTEGER_BINARY | _FLOAT_BINARY, 2), 'fma': 3, 'infinity': 0}


def scalar_op(op, values, target, sources=None):
    """Execute a typed Batch 1 scalar op or construct its shared symbolic node."""
    if op not in _SCALAR_ARITY or len(values) != _SCALAR_ARITY[op]:
        raise Unsupported(f'{op}: 未支持的标量运算或参数数量')
    result = dtype(target)
    source = tuple(map(dtype, sources if sources is not None else map(scalar_type, values)))
    if len(source) != len(values):
        raise Unsupported(f'{op}: 源类型数量与参数不同')

    integer = op in _INTEGER_UNARY or op in _INTEGER_BINARY
    predicates = {'isnan', 'isfinite'}
    if op == 'infinity':
        if result.name not in _FLOAT_FORMATS:
            raise Unsupported('infinity 需要可表示无穷大的浮点类型')
        return cast(math.inf, target)
    if op == 'bitcast':
        if _width(result) != _width(source[0]):
            raise Unsupported('bitcast/reinterpret 需要相同的源和目标位宽')
    elif integer:
        if result.kind not in 'biu' or any(d.kind not in 'biu' for d in source):
            raise Unsupported(f'{op}: 需要标量整数位类型')
        bit_types = source[:1] if op in {'shl', 'ashr', 'lshr'} else source
        if op not in {'ctpop', 'ctlz'} and any(_width(d) != _width(result) for d in bit_types):
            raise Unsupported(f'{op}: 源与目标位宽不同')
    else:
        if not all(map(_float_type, source)) or (result.kind != 'b' if op in predicates else not _float_type(result)):
            raise Unsupported(f'{op}: 源或目标浮点类型不受支持')
        if op == 'fma' and (result.name not in _FLOAT_FORMATS or any(d != result for d in source)):
            raise Unsupported('fma 仅支持相同类型的 f16/bf16/f32/f64 标量；FP8 融合舍入未实现')
        if op in {'copysign', 'ieee_div', 'powf', 'fmod'} and any(d != result for d in source):
            raise Unsupported(f'{op} 需要同类型浮点标量')

    normalized = tuple(value if isinstance(value, Expr) else
                       (_integer_result(int(value), kind) if kind.kind in 'biu'
                        else cast(value, kind.name))
                       for value, kind in zip(values, source))
    if op in {'shl', 'ashr', 'lshr'} and not isinstance(normalized[1], Expr):
        amount = int(normalized[1])
        if not 0 <= amount < _width(result):
            raise SemanticFault('移位量超出目标位宽', rule_id='ARITH-INVALID-SHIFT',
                                shift=amount, bitwidth=_width(result))
    if op in {'sdiv', 'udiv', 'urem'} and not isinstance(normalized[1], Expr):
        divisor = integer_interpret(normalized[1], source[1].name, unsigned=op != 'sdiv')
        if not divisor:
            raise SemanticFault('整数除法或取余除零', rule_id='ARITH-INVALID-DIVISION')
    if any(isinstance(v, Expr) for v in normalized):
        nodes = tuple(literal(v) for v in normalized)
        # Arithmetic and bitwise AND/OR coincide for scalar i1.
        expr_op = {'bitand': 'and', 'bitor': 'or'}.get(op, op) if result.kind == 'b' else ('div' if op == 'ieee_div' else op)
        return Expr(expr_op, nodes, storage_type(result), merge_trace(nodes), result.name)

    if integer:
        a = integer_interpret(normalized[0], source[0].name, unsigned=op not in {'ashr', 'sdiv', 'smin', 'smax', 'ipow'})
        if op in _INTEGER_UNARY:
            value = a.bit_count() if op == 'ctpop' else _width(source[0])-a.bit_length() if op == 'ctlz' else ~a
            return _integer_result(value, result)
        b = integer_interpret(normalized[1], source[1].name, unsigned=op not in {'sdiv', 'smin', 'smax', 'ipow'})
        if op == 'sdiv':
            if a == -(1 << (_width(result) - 1)) and b == -1:
                raise SemanticFault('有符号最小值除以 -1 溢出', rule_id='ARITH-INVALID-DIVISION')
            output = (abs(a) // abs(b)) * (-1 if (a < 0) != (b < 0) else 1)
        elif op in {'ipow', 'upow'}:
            if b < 0:
                if not a:
                    raise SemanticFault('零的负整数次幂', rule_id='ARITH-INVALID-DIVISION')
                # Only 1 and -1 have nonzero integer reciprocals.
                output = 1 if a == 1 else (-1 if b % 2 else 1) if a == -1 else 0
            else:
                output = pow(a, b, 1 << _width(result))
        else:
            output = {'bitand': lambda: a & b, 'bitor': lambda: a | b, 'bitxor': lambda: a ^ b,
                      'shl': lambda: a << b, 'ashr': lambda: a >> b, 'lshr': lambda: a >> b,
                      'udiv': lambda: a // b, 'urem': lambda: a % b,
                      'smin': lambda: min(a, b), 'umin': lambda: min(a, b),
                      'smax': lambda: max(a, b), 'umax': lambda: max(a, b)}[op]()
        return _integer_result(output, result)
    if op == 'bitcast':
        return np.asarray(normalized[0], dtype=source[0]).view(result)[()]
    if op == 'fma':
        return _fused_multiply_add(*normalized, result)
    if op == 'copysign':
        # Preserve the magnitude operand's NaN payload as well as signed zeros.
        bits = np.dtype(f'uint{result.itemsize * 8}')
        magnitude, sign_source = (int(np.asarray(v).view(bits)[()]) for v in normalized)
        sign = 1 << (_width(result) - 1)
        return _from_bits((magnitude & (sign - 1)) | (sign_source & sign), result)
    with np.errstate(all='ignore'):
        fn = _FLOAT_UNARY.get(op) or {'fmod': np.fmod, 'powf': np.power, 'ieee_div': np.divide}[op]
        return cast(fn(*normalized), target)


def tir_scalar(operation, values, target, sources=None):
    """Apply TIR's declared types and the literal control arguments of its calls."""
    op = TIR_BATCH_SCALARS[operation]
    if op == 'infinity':
        # The RISC-V lowerer accepts either no operands or one StringImm hint.
        # The Call's result dtype, not the hint, determines the constant bits.
        if len(values) > 1 or (values and not isinstance(values[0], str)):
            raise Unsupported(f'{operation} 仅接受可选的类型字符串')
        return scalar_op(op, [], target)
    if op in {'ieee_div', 'fma'}:
        if len(values) != _SCALAR_ARITY[op] + 1 or not isinstance(values[-1], str) or values[-1] != 'rn':
            raise Unsupported(f'{operation} 仅支持字面量舍入模式 rn')
        if dtype(target).name not in {'float32', 'float64'}:
            raise Unsupported(f'{operation} 仅支持 float32/float64')
        values = values[:-1]
        sources = None if sources is None else sources[:-1]
    kinds = tuple(sources if sources is not None else map(scalar_type, values))
    if op == 'ashr' and kinds and dtype(kinds[0]).kind == 'u':
        op = 'lshr'
    if op == 'powf' and len(kinds) == 2 and not all(_float_type(dtype(k)) for k in kinds):
        if all(dtype(k).kind in 'iu' for k in kinds):
            if any(dtype(k) != dtype(target) for k in kinds):
                raise Unsupported('tir.pow 的整数底数、指数和结果需要同类型')
            op = 'upow' if dtype(target).kind == 'u' else 'ipow'
        else:
            raise Unsupported('tir.pow 的混合浮点底数/整数指数需要 math.fpowi 语义')
    return scalar_op(op, values, target, kinds)


def mlir_cast(operation, value, source, target):
    """Typed MLIR casts, including checked truncation from float to integer."""
    if operation in {'fptosi', 'fptoui'}:
        src, dst = dtype(source), dtype(target)
        if not _float_type(src) or dst.kind not in 'biu' or target == 'index':
            raise Unsupported(f'{operation} 需要标量浮点源和整型目标')
        if isinstance(value, Expr):
            return Expr(operation, (value,), storage_type(dst), value.trace, dst.name)
        raw = float(cast(value, src.name))
        width = _width(dst)
        signed = operation == 'fptosi'
        lo,hi = (-(1 << (width-1)), (1 << (width-1))-1) if signed else (0,(1 << width)-1)
        truncated = math.trunc(raw) if math.isfinite(raw) else None
        if truncated is None or not lo <= truncated <= hi:
            raise SemanticFault(f'{operation}：输入非有限值或截断结果超出范围',
                                rule_id='ARITH-INVALID-CAST', operation=operation,
                                target_bits=width)
        return _integer_result(truncated, dst)
    if operation == 'bitcast':
        return scalar_op('bitcast', [value], target, [source])
    if operation in {'extui','extsi','uitofp','sitofp','index_cast','index_castui'}:
        value = integer_interpret(value, source, unsigned=operation in {'extui','uitofp','index_castui'})
    elif operation == 'trunci':
        # Truncation keeps low bits, independent of their signed interpretation.
        if dtype(target).kind == 'b':
            if isinstance(value, Expr):
                return Expr('trunc_bits', (value,), 'i1', value.trace)
            return np.bool_(int(value) & 1)
    elif operation not in {'extf','truncf'}:
        raise Unsupported(f'未支持类型转换 {operation}')
    return cast(value, target)


def mlir_cmpi(predicate, left, right, source):
    if predicate in {'eq','ne'} and any(isinstance(v,Expr) for v in (left,right)):
        return apply(predicate,[left,right])
    if predicate in {'eq','ne'}:
        # Equality compares bit patterns; unsigned also handles i1 exactly.
        signed = False
        operation = predicate
    elif predicate in {'slt','sle','sgt','sge','ult','ule','ugt','uge'}:
        signed = predicate[0] == 's'
        operation = predicate[1:]
    else:
        raise Unsupported(f'未知整数比较谓词 {predicate}')
    values = [integer_interpret(v,source,unsigned=not signed) for v in (left,right)]
    if not any(isinstance(v, Expr) for v in values):
        # Mixed Python integers near 2**63 can make NumPy select float64,
        # losing low bits. Integer comparisons must remain exact.
        a,b = values
        return np.bool_({'eq':lambda:a==b,'ne':lambda:a!=b,'lt':lambda:a<b,
                         'le':lambda:a<=b,'gt':lambda:a>b,'ge':lambda:a>=b}[operation]())
    return apply(operation, values)


MLIR_CMPF_PREDICATES = ('false','oeq','ogt','oge','olt','ole','one','ord',
                        'ueq','ugt','uge','ult','ule','une','uno','true')


def mlir_cmpf(predicate, left, right):
    if predicate not in MLIR_CMPF_PREDICATES:
        raise Unsupported(f'未知浮点比较谓词 {predicate}')
    if isinstance(left, Expr) or isinstance(right, Expr):
        nodes = (literal(left), literal(right))
        return Expr('cmpf_'+predicate, nodes, 'i1', merge_trace(nodes))
    unordered = bool(np.isnan(left) or np.isnan(right))
    special = {'false':False, 'true':True, 'ord':not unordered, 'uno':unordered}
    if predicate in special:
        return np.bool_(special[predicate])
    result = apply(predicate[1:], [left,right])
    return np.bool_(unordered or result if predicate[0]=='u' else not unordered and result)


def select(condition, yes, no):
    if isinstance(condition, Expr):
        a, b = literal(yes), literal(no)
        if a.type != b.type:
            raise Unsupported("选择分支类型不同")
        return Expr("select", (condition, a, b), a.type, merge_trace((condition, a, b)),
                    a.scalar_dtype if a.scalar_dtype == b.scalar_dtype else None)
    return yes if bool(condition) else no


class Budget:
    def __init__(self, max_steps=2_000_000, max_elements=1_000_000):
        self.max_steps, self.max_elements = max_steps, max_elements
        self.steps = 0
        self.allocated_elements = 0
        self.parallel=[]
        self.region_count=0
        self.context=[]

    @contextmanager
    def at(self, kind, line, **values):
        self.context.append({'kind': kind, 'line': line, **values})
        try:
            yield
        finally:
            self.context.pop()

    def new_region(self):
        self.region_count+=1
        return self.region_count

    @contextmanager
    def iteration(self,region,index):
        self.parallel.append((region,index))
        try:
            yield
        finally:
            self.parallel.pop()

    def tick(self, n=1):
        self.steps += n
        if self.steps > self.max_steps:
            raise Unsupported(f"超出执行步数上限 {self.max_steps}")

    def shape(self, shape):
        if any(type(int(n)) is not int or int(n) < 0 for n in shape):
            raise Unsupported("非法形状")
        if np.prod(shape, dtype=object) > self.max_elements:
            raise Unsupported(f"单个缓冲区超出元素上限 {self.max_elements}")


class Buffer:
    def __init__(self, values, *, initialized, symbolic, position=None, budget=None,
                 name=None, allocation_line=None, readonly=False):
        budget=budget or Budget()
        budget.allocated_elements+=np.asarray(values).size
        if budget.allocated_elements>8*budget.max_elements:
            raise Unsupported('累计缓冲区分配超出元素预算，拒绝继续分配')
        self.values = np.array(values, copy=True, order="C")
        if not initialized:
            self.values.fill(0)
        self.dtype = self.values.dtype
        self.shape = self.values.shape
        self.root = self
        self.element_offset = 0
        # NumPy may report all-zero strides for empty allocations. The memref
        # descriptor still has the logical strides of an identity layout.
        self.element_strides = tuple(math.prod(self.shape[i+1:]) for i in range(len(self.shape)))
        self.budget = budget
        self.budget.shape(self.shape)
        self.initialized = np.full(self.shape, initialized, dtype=bool)
        self.writes = np.zeros(self.shape, dtype=np.int64)
        self.last_lines = np.zeros(self.shape, dtype=np.int64)
        self.symbolic = symbolic
        self.accesses={}
        self.name = name or (f'arg{position}' if position is not None else 'temporary')
        self.position, self.allocation_line, self.readonly = position, allocation_line, readonly
        self.expressions = None
        if symbolic:
            self.expressions = np.empty(self.shape, dtype=object)
            for index in np.ndindex(self.shape):
                self.expressions[index] = (Expr("input", (position, index), storage_type(self.dtype), scalar_dtype=self.dtype.name)
                                           if initialized else None)

    def subview(self, offsets, sizes, strides, dropped=()):
        offsets, sizes, strides = (tuple(map(concrete_index,part))
                                   for part in (offsets,sizes,strides))
        if not (len(offsets)==len(sizes)==len(strides)==len(self.shape)):
            raise Unsupported("subview rank 不同")
        for a,n,k,b in zip(offsets,sizes,strides,self.shape):
            if a<0 or n<0 or k<=0 or a+n*k-k>=b and n:
                self.fault('MEM-OUT-OF-BOUNDS','subview 区域越界',0,tuple(offsets))
        if any(sizes[i]!=1 for i in dropped):
            raise Unsupported("只能降掉 size=1 的 subview 维度")
        offset = self.element_offset + sum(a*s for a,s in zip(offsets,self.element_strides))
        shape = tuple(n for i,n in enumerate(sizes) if i not in dropped)
        steps = tuple(s*k for i,(s,k) in enumerate(zip(self.element_strides,strides)) if i not in dropped)
        return self._shared_view(offset, shape, steps)

    def reinterpret_cast(self, offset, sizes, strides, *, element_type=None, line=0):
        """Build a descriptor relative to the root allocation, not this view."""
        if element_type is not None and storage_type(dtype(element_type)) != storage_type(self.dtype):
            raise Unsupported('reinterpret_cast 不能改变元素类型')
        offset, sizes, strides = (concrete_index(offset),
                                  tuple(map(concrete_index, sizes)),
                                  tuple(map(concrete_index, strides)))
        if len(sizes) != len(strides) or offset < 0 or any(n < 0 for n in sizes) or any(s <= 0 for s in strides):
            raise Unsupported('reinterpret_cast 需要非负 offset/size、正 stride 及匹配的 rank')
        self.budget.shape(sizes)
        if 0 not in sizes:
            last = offset + sum((n-1)*s for n,s in zip(sizes,strides))
            if last >= self.root.values.size:
                self.fault('MEM-OUT-OF-BOUNDS', 'reinterpret_cast 超出根缓冲区', line, (last,),
                           root_elements=self.root.values.size, element_offset=offset)
        return self._shared_view(offset, sizes, strides)

    def _shared_view(self, offset, sizes, strides):
        result=object.__new__(Buffer)
        result.__dict__=dict(self.__dict__)
        # An empty view accesses no address. Preserve its logical descriptor,
        # but do not form an out-of-allocation pointer in the NumPy adapter.
        physical_offset = 0 if 0 in sizes else offset
        for attr in ['values','initialized','writes','last_lines','expressions']:
            arr=getattr(self.root,attr)
            try:
                view = None if arr is None else np.ndarray(
                    sizes, dtype=arr.dtype, buffer=arr,
                    offset=physical_offset * arr.itemsize,
                    strides=tuple(s * arr.itemsize for s in strides))
            except (ValueError, OverflowError) as exc:
                raise Unsupported('不能构造该 memref 共享视图') from exc
            setattr(result,attr,view)
        result.shape, result.element_offset, result.element_strides = sizes, offset, strides
        return result

    def root_index(self, index):
        if self is self.root:
            return tuple(map(int,index))
        flat = self.element_offset + sum(int(i)*s for i,s in zip(index,self.element_strides))
        return tuple(map(int,np.unravel_index(flat,self.root.shape)))

    def fault(self, rule_id, message, line, index, **details):
        raise SemanticFault(f'L{line}: {message}', rule_id=rule_id, line=line,
            buffer=self.name, parameter=self.position, shape=list(self.shape),
            index=list(index), allocation_line=self.allocation_line,
            context=list(self.budget.context), **details)

    def trace(self, value, line, action, index):
        return trace_value(value, self.name, line, action, index)

    def index(self, index, line, access='read'):
        if not isinstance(index, tuple):
            index = (index,)
        if len(index) != len(self.shape) or any(isinstance(i, Expr) for i in index):
            raise Unsupported(f"L{line}: 不支持的内存下标 {index}")
        index = tuple(int(i) for i in index)
        if any(i < 0 or i >= size for i, size in zip(index, self.shape)):
            self.fault('MEM-OUT-OF-BOUNDS', f'内存访问越界 {index}，shape={self.shape}', line, index, access=access)
        return index

    def read(self, index, line):
        self.budget.tick()
        index = self.index(index, line)
        self.access(index,False,line)
        if not self.initialized[index]:
            self.fault('MEM-UNINITIALIZED', f'读取未初始化数据 {index}', line, index, access='read')
        return self.trace(self.expressions[index], line, 'read', index) if self.symbolic else self.values[index]

    def write(self, index, value, line):
        self.budget.tick()
        index = self.index(index, line, 'write')
        if self.readonly:
            self.fault('MEM-READONLY-WRITE', '修改了只读输入参数', line, index, access='write')
        self.access(index,True,line)
        if self.symbolic:
            self.expressions[index] = self.trace(cast(literal(value), str(self.dtype)), line, 'write', index)
        else:
            self.values[index] = value
        self.initialized[index] = True
        self.writes[index] += 1
        self.last_lines[index] = line

    def access(self,index,write,line):
        if not self.budget.parallel:
            return
        if self is not self.root:
            return self.root.access(self.root_index(index),write,line)
        for region,iteration in self.budget.parallel:
            record=self.accesses.setdefault(region,{}).setdefault(index,{'writer':None,'readers':set(), 'last_line':None})
            if (record['writer'] is not None and record['writer']!=iteration
                    or write and any(reader!=iteration for reader in record['readers'])):
                self.fault('PARALLEL-CONFLICT', f'逻辑 parallel 迭代之间存在读写/写写冲突，下标 {index}',
                    line, index, conflicting_line=record['last_line'], parallel_region=region,
                    previous_iteration=record['writer'], current_iteration=iteration)
            if write:
                record['writer']=iteration
            else:
                record['readers'].add(iteration)
            record['last_line']=line

    def summary(self):
        return {"buffer": self.name, "allocation_line": self.allocation_line,
                "shape": list(self.shape), "dtype": str(self.dtype),
                "elements": int(self.values.size),
                "unwritten": int(np.count_nonzero(self.writes == 0)),
                "multiply_written": int(np.count_nonzero(self.writes > 1)),
                "writer_lines": sorted(set(int(v) for v in self.last_lines.flat if v))}


def concrete_index(value):
    if isinstance(value,Expr):
        raise Unsupported('尚不支持数据相关符号下标、尺寸或步长；可单独执行具体输入')
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise Unsupported('需要具体整数下标、尺寸或步长')
    return int(value)


def make_buffers(arguments, outputs, symbolic, budget, inouts=()):
    result = []
    for index, value in enumerate(arguments):
        if not isinstance(value, np.ndarray) or not value.flags.c_contiguous:
            raise Unsupported("当前执行后端要求连续 NumPy 数组参数")
        result.append(Buffer(value, initialized=index not in outputs or index in inouts, symbolic=symbolic,
                             position=index, budget=budget, readonly=index not in outputs and index not in inouts))
        result[-1].require_full_write=index not in inouts
    for i, a in enumerate(arguments):
        for b in arguments[:i]:
            if np.shares_memory(a, b):
                raise Unsupported("当前执行契约要求参数之间不别名")
    return result


def expression_digest(array):
    h = hashlib.sha256()
    for value in array.flat:
        h.update(value.fingerprint().encode() if value is not None else b"null")
        h.update(b"\n")
    return h.hexdigest()
