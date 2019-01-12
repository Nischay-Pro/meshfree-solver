import numpy as np
cimport numpy as np 

cdef np.ndarray outer_dGx_pos(list globaldata, int idx, dict configData)
cdef np.ndarray outer_dGx_neg(list globaldata, int idx, dict configData)
cdef np.ndarray outer_dGy_pos(list globaldata, int idx, dict configData)