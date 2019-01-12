import numpy as np
cimport numpy as np 

cdef np.ndarray venkat_limiter(np.ndarray qtilde, list globaldata, int idx, dict configData)
cdef max_q_values(list globaldata, int idx)
cdef min_q_values(list globaldata, int idx)