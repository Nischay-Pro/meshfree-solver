# cython: profile=True
# cython: binding=True
# cython: linetrace=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1
from libc.math cimport sqrt, pow
import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False) 
@cython.cdivision(True)
cdef np.ndarray venkat_limiter(np.ndarray qtilde, list globaldata, int idx, dict configData):
    cdef int VL_CONST = configData["core"]["vl_const"]
    cdef np.ndarray phi = np.zeros((4), dtype=np.float64)
    cdef double del_pos = 0
    cdef double del_neg = 0
    cdef int i = 0
    cdef double q = 0
    cdef double max_q, ds, epsi, num, den, temp, min_q
    cdef np.ndarray currq = globaldata[idx].getq()
    for i in range(4):
        q = currq[i]
        del_neg = qtilde[i] - q
        if abs(del_neg) <= 1e-5:
            phi[i] = 1
        elif abs(del_neg) > 1e-5:
            if del_neg > 0:
                max_q = maximum(globaldata, idx, i, currq)
                del_pos = max_q - q
            elif del_neg < 0:
                min_q = minimum(globaldata, idx, i, currq)
                del_pos = min_q - q

            ds = smallest_dist(globaldata,idx)
            epsi = VL_CONST * ds
            epsi = pow(epsi,3)

            num = (del_pos*del_pos) + (epsi*epsi)
            num = num*del_neg + 2.0*del_neg*del_neg*del_pos

            den = del_pos*del_pos + 2.0*del_neg*del_neg
            den = den + del_neg*del_pos + epsi*epsi
            den = den*del_neg

            temp = num/den

            if temp < 1:
                phi[i] = temp
            else:
                phi[i] = 1
    return phi


@cython.boundscheck(False)
@cython.wraparound(False) 
cdef maximum(list globaldata, int idx, int i, np.ndarray currq):
    cdef double maxval
    cdef int itm
    maxval = currq[i]
    for itm in globaldata[idx].get_conn():
        if maxval < globaldata[itm].getq()[i]:
            maxval = globaldata[itm].getq()[i]
    return maxval

@cython.boundscheck(False)
@cython.wraparound(False) 
cdef minimum(list globaldata, int idx, int i, np.ndarray currq):
    cdef double minval
    cdef int itm
    minval = currq[i]
    for itm in globaldata[idx].get_conn():
        if minval > globaldata[itm].getq()[i]:
            minval = globaldata[itm].getq()[i]
    return minval

@cython.boundscheck(False)
@cython.wraparound(False) 
cdef smallest_dist(list globaldata, int idx):
    cdef double min_dist = 10000
    cdef int itm
    cdef double dx, dy, ds
    for itm in globaldata[idx].get_conn():
        dx = globaldata[idx].getx() - globaldata[itm].getx()
        dy = globaldata[idx].gety() - globaldata[itm].gety()
        ds = sqrt(dx * dx + dy * dy)
        
        if ds < min_dist:
            min_dist = ds

    return min_dist

@cython.boundscheck(False)
@cython.wraparound(False) 
cdef max_q_values(list globaldata, int idx):
    cdef double [:] maxq
    cdef int i = 0
    cdef int itm
    cdef double [:] currq
    maxq = globaldata[idx].getq()
    for itm in globaldata[idx].get_conn():
        currq = globaldata[itm].getq()
        for i in range(4):
            if maxq[i] < currq[i]:
                maxq[i] = currq[i]
    return maxq

@cython.boundscheck(False)
@cython.wraparound(False) 
cdef min_q_values(list globaldata, int idx):
    cdef double [:] minq
    cdef double [:] currq
    cdef int itm
    cdef int i = 0
    minq = globaldata[idx].getq()
    for itm in globaldata[idx].get_conn():
        currq = globaldata[itm].getq()
        for i in range(4):
            if minq[i] > currq[i]:
                minq[i] = currq[i]
    return minq