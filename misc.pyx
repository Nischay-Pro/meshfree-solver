import numpy as np
cimport numpy as np
from libc.math cimport log, exp

cdef np.ndarray qtilde_to_primitive(np.ndarray qtilde, dict configData):
    
    cdef double gamma = configData["core"]["gamma"]

    cdef np.ndarray[np.float64_t, ndim=1] result = np.zeros(4)

    cdef double q1 = qtilde[0]
    cdef double q2 = qtilde[1]
    cdef double q3 = qtilde[2]
    cdef double q4 = qtilde[3]

    cdef double beta = -q4*0.5

    cdef double temp = 0.5/beta

    cdef double u1 = q2*temp
    cdef double u2 = q3*temp

    cdef double temp1 = q1 + beta*(u1*u1 + u2*u2)
    cdef double temp2 = temp1 - (log(beta)/(gamma-1))
    cdef double rho = exp(temp2)
    cdef double pr = rho*temp

    result[0] = u1
    result[1] = u2
    result[2] = rho
    result[3] = pr

    return result