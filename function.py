import numpy as np
from itertools import product as cross
def _append(arr1, arr2, axis=None):
    return arr2 if arr1 is None else np.append(arr1, arr2, axis)
        
def _conv_single_data_single_filter(m, w):
    H,W = m.shape[:-1]
    kernel_H, kernel_W = w.shape[:-1]
    result = np.zeros((H-kernel_H+1, W-kernel_W+1, 1), dtype=float)
    for h, w in cross(range(H-kernel_H+1), range(W-kernel_W+1)):
        block = m[h:h+kernel_H, w:w+kernel_W, :]
        product = (block * w).sum()
        result[h,w,0] = product
    return result

# w [n_filter, h, w, c]
def _conv_single_data_multi_filters(m, w):
    result = None
    for i_filter in range(w.shape[0]):
        result = _append(result, _conv_single_data_single_filter(m, w[i_filter]), axis=-1)
    return result

def _conv_multi_data_multi_filters(m, w):
    result = None
    for i_batch in range(m.shape[0]):
        result = _append(result, _conv_single_data_multi_filters(m[i_batch], w), axis=0)
    return result
    
def conv(m, w):
    # TODO padding and strides
    return _conv_multi_data_multi_filters(m, w)

def rotate(w):
    return w[:,::-1,::-1,:]

def pad(m, paddings=[0,0]):
    # m is conv data
    paddings_axes = [ [0,0], [paddings[0], paddings[0]], [paddings[1], paddings[1]], [0,0] ]
    return np.pad(m, paddings_axes, 'constant', constant_values=0)

def rearange(w):
    return np.transpose(w, [3,1,2,0])

def trans_coord_from_1d_to_2d(index_1d, w):
    # index_1d [batch, channel]
    index_1d = index_1d[...,np.newaxis]
    return np.concatenate((index_1d/w, index_1d%w), axis=-1)

if __name__ == '__main__':
    a = np.arange(24).reshape(2,2,3,2)
    print a
    print '--------------------'
    print rotate(a)
