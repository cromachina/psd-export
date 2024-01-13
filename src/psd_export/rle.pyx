#cython: language_level=3, boundscheck=False, wraparound=False
import sys
from libc.string cimport memcpy, memset
from libc.stdint cimport *
import numpy as np
import psd_tools.api.numpy_io as numpy_io
import psd_tools.constants as ptc
from psd_tools.api.layers import Layer

# A more speed optimized RLE decode based on https://github.com/psd-tools/psd-tools/blob/main/src/psd_tools/compression/
cdef decode(const uint32_t[:] counts, const uint8_t[:] rows, size_t height, size_t width):
    cdef uint32_t row_offset = 0, count
    cdef int src, dst, length, header, i
    cdef const uint8_t[:] row_view
    result = np.empty((height, width), dtype=np.ubyte)
    cdef uint8_t[:,:] result_view = result
    cdef uint8_t[:] result_row

    with nogil:
        for i in range(counts.shape[0]):
            count = counts[i]
            row_view = rows[row_offset:row_offset + count]
            result_row = result_view[i]
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
    return decode(counts.astype(np.uint32), rows, height, row_size)

def layer_numpy(layer:Layer, channel=None):
    if channel == 'mask' and not layer.mask:
        return None

    depth = layer._psd.depth
    version = layer._psd.version

    def channel_matches(info):
        if channel == 'color':
            return info.id >= 0
        if channel == 'shape':
            return info.id == ptc.ChannelID.TRANSPARENCY_MASK
        if channel == 'mask':
            if not layer.mask:
                return False
            if layer.mask._has_real():
                return info.id == ptc.ChannelID.REAL_USER_LAYER_MASK
            else:
                return info.id == ptc.ChannelID.USER_LAYER_MASK
        else:
            raise ValueError(f'Unknown channel type: {channel}')

    channels = zip(layer._channels, layer._record.channel_info)
    channels = [channel for channel, info in channels if channel_matches(info)]

    if len(channels) == 0:
        return None

    # Use the psd-tools path if we are not decoding RLE
    if not all([channel.compression == ptc.Compression.RLE for channel in channels]):
        return layer.numpy(channel)

    if channel == 'mask':
        width, height = layer.mask.width, layer.mask.height
    else:
        width, height = layer.width, layer.height

    decoded = []
    for channel in channels:
        data = decode_rle(channel.data, width, height, depth, version)
        data = numpy_io._parse_array(data, depth)
        decoded.append(data)

    return np.stack(decoded, axis=1).reshape((height, width, -1))
