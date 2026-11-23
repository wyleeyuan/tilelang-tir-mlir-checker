"""Bounded, object-only replay contracts for the fixed RISC-V lowerer."""
from dataclasses import dataclass
from math import prod

from .execution import SemanticFault, Unsupported, cast, concrete_index, scalar_op


ADDRESS = 'tir.address_of'
ACCESS = 'tir.tvm_access_ptr'
PTYPE = 'tir.type_annotation'
ATOMIC_LOAD = 'tl.atomic_load_elem_op'
ATOMIC_STORE = 'tl.atomic_store_elem_op'
STREAM_VECTOR = frozenset('tileops::'+name for name in ('load16','load16_evict_first','load16_evict_last'))
STREAM_SCALAR = frozenset('tileops::'+name for name in ('load_evict_first','load_evict_last','load_f32_evict_last'))
SHUFFLES = {
    'tl.shfl_sync': 'idx', 'tir.tvm_warp_shuffle': 'idx',
    'tl.shfl_xor_sync': 'xor',
    'tl.shfl_down_sync': 'down', 'tir.tvm_warp_shuffle_down': 'down',
    'tl.shfl_up_sync': 'up', 'tir.tvm_warp_shuffle_up': 'up',
}
INT = frozenset(prefix+str(bits) for prefix in ('int','uint') for bits in (8,16,32,64))
SCALAR = INT | {'float16','float32','float64','bfloat16','float8_e4m3fn'}


@dataclass(frozen=True)
class PointerView:
    """A window of logical contiguous elements; all accesses go through Buffer."""
    buffer: object
    offset: int
    extent: int
    mask: int

    def __post_init__(self):
        off, count, mask = map(concrete_index, (self.offset,self.extent,self.mask))
        if not 1 <= mask <= 3:
            raise Unsupported('有界引用需要合法 rw_mask（1/2/3）')
        buf = self.buffer
        if not buf.shape or buf.element_strides != tuple(prod(buf.shape[i+1:]) for i in range(len(buf.shape))):
            raise Unsupported('有界引用仅支持紧凑行主序 Buffer')
        if off < 0 or count < 0 or off + count > prod(buf.shape):
            buf.fault('MEM-OUT-OF-BOUNDS','有界引用超出 Buffer',0,(off,),extent=count)

    def index(self, element, access):
        element = concrete_index(element)
        if not self.mask & {'read':1,'write':2}[access]:
            raise Unsupported(f'有界引用 rw_mask 不允许 {access}')
        if not 0 <= element < self.extent:
            self.buffer.fault('MEM-OUT-OF-BOUNDS','访问超出有界引用范围',0,(element,),extent=self.extent)
        remaining = self.offset + element
        index = []
        for stride in self.buffer.element_strides:
            coord,remaining = divmod(remaining,stride)
            index.append(coord)
        return tuple(index)

    def read(self, element):
        return self.buffer.read(self.index(element,'read'),0)


def stream_words(src, target):
    """LowerStreamingVectorLoadCallExtern: gather 16 bytes, little endian."""
    src_type = src.buffer.dtype.name
    sw, tw = src.buffer.dtype.itemsize, target.buffer.dtype.itemsize
    target_type = target.buffer.dtype.name
    values = [src.read(i) for i in range(16//sw)]
    if src_type == target_type:
        return values
    su,tu = f'uint{8*sw}', f'uint{8*tw}'
    words = [scalar_op('bitcast',[v],su,[src_type]) for v in values]
    result = []
    for byte in range(0,16,tw):
        if sw >= tw:
            word = words[byte//sw]
            shift = 8*(byte%sw)
            if shift:
                word = scalar_op('lshr',[word,shift],su,[su,'int32'])
            word = cast(word,tu)
        else:
            word = cast(words[byte//sw],tu)
            for i in range(1,tw//sw):
                part = cast(words[byte//sw+i],tu)
                part = scalar_op('shl',[part,8*i*sw],tu,[tu,'int32'])
                word = scalar_op('bitor',[word,part],tu,[tu,tu])
        result.append(scalar_op('bitcast',[word],target_type,[tu]))
    return result


def shuffle_source(mode, tid, extent, arg, width, mask, mask_bits=32):
    lane = tid % 32
    base = lane // width * width
    sub = lane - base
    if mode == 'up' and sub < arg:
        selected, valid = 0, False  # lowerer selects zero before forming the mask shift
    else:
        selected = {'idx':lambda: arg%width, 'xor':lambda: sub^arg,
                    'down':lambda: sub+arg, 'up':lambda: sub-arg}[mode]()
        valid = selected < width
    source_lane = base + selected
    source = tid-lane+source_lane
    if not 0 <= source_lane < mask_bits:
        raise SemanticFault('Warp shuffle: 源 lane 导致降级后的 mask 左移超出位宽',
                            rule_id='ARITH-INVALID-SHIFT', shift=source_lane, bitwidth=mask_bits)
    if valid and source_lane < 32 and source < extent and mask & (1 << source_lane):
        return source
    return tid
