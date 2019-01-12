import numpy as np
cimport numpy as np

cdef np.ndarray flux_Gxp(double nx, double ny, double u1, double u2, double rho, double pr)
cdef np.ndarray flux_Gxn(double nx, double ny, double u1, double u2, double rho, double pr)
cdef np.ndarray flux_Gyp(double nx, double ny, double u1, double u2, double rho, double pr)
cdef np.ndarray flux_Gyn(double nx, double ny, double u1, double u2, double rho, double pr)