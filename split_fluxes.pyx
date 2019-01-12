# cython: profile=True
# cython: binding=True
# cython: linetrace=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1
from libc.math cimport sqrt, erf, exp, acos
import numpy as np 
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray flux_Gxp(double nx, double ny, double u1, double u2, double rho, double pr):

    cdef double pi = acos(-1)
    cdef np.ndarray[np.float64_t, ndim=1] Gxp = np.zeros(4)

    cdef double tx = ny
    cdef double ty = -nx

    cdef double ut = u1*tx + u2*ty
    cdef double un = u1*nx + u2*ny

    cdef double beta = 0.5*rho/pr
    cdef double S1 = ut*sqrt(beta) 
    cdef double B1 = 0.5*exp(-S1*S1)/sqrt(pi*beta)
    cdef double A1pos = 0.5*(1 + erf(S1))     

    cdef double pr_by_rho = pr/rho
    cdef double u_sqr = ut*ut + un*un

    Gxp[0] = rho*(ut*A1pos + B1)
        
    cdef double temp1 = pr_by_rho + ut*ut
    cdef double temp2 = temp1*A1pos + ut*B1
    Gxp[1] = rho*temp2

    temp1 = ut*un*A1pos + un*B1
    Gxp[2] = rho*temp1

    temp1 = (7*pr_by_rho) + u_sqr
    temp2 = 0.5*ut*temp1*A1pos 
    temp1 = (6*pr_by_rho) + u_sqr
    Gxp[3] = rho*(temp2 + 0.5*temp1*B1)
    return Gxp

@cython.boundscheck(False)
@cython.wraparound(False) 
cdef np.ndarray flux_Gxn(double nx, double ny, double u1, double u2, double rho, double pr):

    cdef double pi = acos(-1)
    cdef np.ndarray Gxn = np.zeros(4)

    cdef double tx = ny
    cdef double ty = -nx

    cdef double ut = u1*tx + u2*ty
    cdef double un = u1*nx + u2*ny

    cdef double beta = 0.5*rho/pr
    cdef double S1 = ut*sqrt(beta) 
    cdef double B1 = 0.5*exp(-S1*S1)/sqrt(pi*beta)
    cdef double A1neg = 0.5*(1 - erf(S1))     

    cdef double pr_by_rho = pr/rho
    cdef double u_sqr = ut*ut + un*un


    Gxn[0] = rho*(ut*A1neg - B1)

    cdef double temp1 = pr_by_rho + ut*ut
    cdef double temp2 = temp1*A1neg - ut*B1
    Gxn[1] = rho*temp2

    temp1 = ut*un*A1neg - un*B1
    Gxn[2] = rho*temp1

    temp1 = (7*pr_by_rho) + u_sqr
    temp2 = 0.5*ut*temp1*A1neg 
    temp1 = (6*pr_by_rho) + u_sqr
    Gxn[3] = rho*(temp2 - 0.5*temp1*B1)

    return Gxn

@cython.boundscheck(False)
@cython.wraparound(False) 
cdef np.ndarray flux_Gyp(double nx, double ny, double u1, double u2, double rho, double pr):

    cdef double pi = acos(-1)
    cdef np.ndarray Gyp = np.zeros(4)

    cdef double tx = ny
    cdef double ty = -nx
    cdef double ut = u1*tx + u2*ty
    cdef double un = u1*nx + u2*ny
    cdef double beta = 0.5*rho/pr
    cdef double S2 = un*sqrt(beta) 
    cdef double B2 = 0.5*exp(-S2*S2)/sqrt(pi*beta)
    cdef double A2pos = 0.5*(1 + erf(S2))     
    cdef double pr_by_rho = pr/rho
    cdef double u_sqr = ut*ut + un*un

    Gyp[0] = rho*(un*A2pos + B2)

    cdef double temp1 = pr_by_rho + un*un
    cdef double temp2 = temp1*A2pos + un*B2

    temp1 = ut*un*A2pos + ut*B2
    Gyp[1] = rho*temp1

    Gyp[2] = rho*temp2

    temp1 = (7*pr_by_rho) + u_sqr
    temp2 = 0.5*un*temp1*A2pos 
    temp1 = (6*pr_by_rho) + u_sqr
    Gyp[3] = rho*(temp2 + 0.5*temp1*B2)
    
    return Gyp

@cython.boundscheck(False)
@cython.wraparound(False) 
cdef np.ndarray flux_Gyn(double nx, double ny, double u1, double u2, double rho, double pr):

    cdef double pi = acos(-1)
    cdef np.ndarray Gyn = np.zeros(4)

    cdef double tx = ny
    cdef double ty = -nx

    cdef double ut = u1*tx + u2*ty
    cdef double un = u1*nx + u2*ny
    cdef double beta = 0.5*rho/pr
    cdef double S2 = un*sqrt(beta) 
    cdef double B2 = 0.5*exp(-S2*S2)/sqrt(pi*beta)
    cdef double A2neg = 0.5*(1 - erf(S2))     
    cdef double pr_by_rho = pr/rho
    cdef double u_sqr = ut*ut + un*un

    Gyn[0] = rho*(un*A2neg - B2)
    
    cdef double temp1 = pr_by_rho + un*un
    cdef double temp2 = temp1*A2neg - un*B2
    Gyn[1] = rho*temp2

    temp1 = ut*un*A2neg - ut*B2
    Gyn[2] = rho*temp1

    temp1 = (7*pr_by_rho) + u_sqr
    temp2 = 0.5*un*temp1*A2neg 
    temp1 = (6*pr_by_rho) + u_sqr
    Gyn[3] = rho*(temp2 - 0.5*temp1*B2)

    return Gyn

# def flux_Gx(Gx, nx, ny, u1, u2, rho, pr):
#     tx = ny
#     ty = -nx

#     ut = u1*tx + u2*ty
#     un = u1*nx + u2*ny

#     Gx[0] = rho*ut

#     Gx[1] = pr + rho*ut*ut

#     Gx[2] = rho*ut*un

#     temp1 = 0.5*(ut*ut + un*un)
#     rho_e = 2.5*pr + rho*temp1
#     Gx[3] = (pr + rho_e)*ut

# def flux_Gy(Gy, nx, ny, u1, u2, rho, pr):
#     tx = ny
#     ty = -nx

#     ut = u1*tx + u2*ty
#     un = u1*nx + u2*ny

#     Gy[0] = rho*un

#     Gy[1] = rho*ut*un

#     Gy[2] = pr + rho*un*un

#     temp1 = 0.5*(ut*ut + un*un)
#     rho_e = 2.5*pr + rho*temp1
#     Gy[3] = (pr + rho_e)*un