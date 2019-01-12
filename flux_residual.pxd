import numpy as np
cimport numpy as np 

cdef list cal_flux_residual(list globaldata, list wallindices, list outerindices, list interiorindices, dict configData)