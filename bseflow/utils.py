import numpy as np
import numba as nb

@nb.njit(parallel=True)
def in1d(arr, arr2):
    
    #numba alternative to np.in1d
    arr2_set = set(arr2)
    out = np.empty(arr.shape[0], dtype=nb.boolean)
    for i in nb.prange(arr.shape[0]):
        out[i] = arr[i] in arr2_set

    return out

@nb.njit(parallel=True)
def sum(arr):
    return np.sum(arr)