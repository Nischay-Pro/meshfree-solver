import numpy as np
cimport numpy as np 

cdef np.ndarray wall_dGx_pos(list globaldata, int idx, dict configData)
cdef np.ndarray wall_dGx_neg(list globaldata, int idx, dict configData)
cdef np.ndarray wall_dGy_neg(list globaldata, int idx, dict configData)