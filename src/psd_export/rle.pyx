#cython: language_level=3, boundscheck=False, wraparound=False
import sys
from libc.string cimport memcpy, memset
from libc.stdint cimport *
import numpy as np

# A more speed optimized RLE decode based on https://github.com/psd-tools/psd-tools/blob/main/src/psd_tools/compression/
cdef decode(const uint32_t[:] counts, const uint8_t[:] rows, uint8_t[:,:] result):
    cdef uint32_t row_offset = 0, count
    cdef int src, dst, length, header, i, src_next, dst_next
    cdef const uint8_t[:] row_view
    cdef uint8_t[:] result_row
    cdef size_t width = result.shape[1]
    cdef size_t data_size

    with nogil:
        for i in range(counts.shape[0]):
            count = counts[i]
            row_view = rows[row_offset:row_offset + count]
            result_row = result[i]
            src = 0
            dst = 0
            data_size = row_view.shape[0]
            while src < data_size:
                header = <int> (<int8_t> (<void*> row_view[src]))
                src = src + 1
                if 0 <= header <= 127:
                    length = header + 1
                    src_next = src + length
                    dst_next = dst + length
                    if src_next <= data_size and dst_next <= width:
                        memcpy(&result_row[dst], &row_view[src], length)
                        src = src_next
                        dst = dst_next
                    else:
                        raise ValueError('Invalid RLE compression')
                elif header == -128:
                    pass
                else:
                    length = 1 - header
                    src_next = src + 1
                    dst_next = dst + length
                    if src_next <= data_size and dst_next <= width:
                        memset(&result_row[dst], row_view[src], length)
                        src = src_next
                        dst = dst_next
                    else:
                        raise ValueError('Invalid RLE compression')
            if dst < width:
                raise ValueError('Expected %d bytes but decoded only %d bytes' % (width, dst))
            row_offset += count
    return result

def decode_rle(data, width, height, depth, version):
    row_size = max(width * depth // 8, 1)
    dtype = (np.uint16, np.uint32)[version - 1]
    counts = np.frombuffer(data, dtype=dtype, count=height).copy()
    if sys.byteorder == 'little':
        counts.byteswap(inplace=True)
    rows = np.frombuffer(data, dtype=np.ubyte, offset=counts.nbytes)
    result = np.empty((height, row_size), dtype=np.ubyte)
    decode(counts.astype(np.uint32), rows, result)
    return result
