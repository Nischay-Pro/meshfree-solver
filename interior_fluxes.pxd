import numpy as np
cimport numpy as np

cdef np.ndarray interior_dGx_pos(list globaldata, int idx, dict configData)
cdef np.ndarray interior_dGx_neg(list globaldata, int idx, dict configData)
cdef np.ndarray interior_dGy_pos(list globaldata, int idx, dict configData)
cdef np.ndarray interior_dGy_neg(list globaldata, int idx, dict configData)