import numpy as np
cimport numpy as np

cdef np.ndarray[np.long] flux_quad_GxI(double nx, double ny, double u1, double u2, double rho, double pr)
cdef np.ndarray[np.long] flux_quad_GxII(double nx, double ny, double u1, double u2, double rho, double pr)
cdef np.ndarray[np.long] flux_quad_GxIII(double nx, double ny, double u1, double u2, double rho, double pr)
cdef np.ndarray[np.long] flux_quad_GxIV(double nx, double ny, double u1, double u2, double rho, double pr)