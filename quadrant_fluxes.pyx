# cython: profile=True
# cython: binding=True
# cython: linetrace=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1
import math
import numpy as np
cimport numpy as np 

cdef np.ndarray[np.long] flux_quad_GxI(double nx, double ny, double u1, double u2, double rho, double pr):
    cdef np.ndarray G = np.zeros(4)
    cdef double tx = ny
    cdef double ty = -nx
    cdef double ut = u1*tx + u2*ty
    cdef double un = u1*nx + u2*ny
    cdef double beta = 0.5*rho/pr
    cdef double S1 = ut*math.sqrt(beta)
    cdef double S2 = un*math.sqrt(beta)
    cdef double B1 = 0.5*math.exp(-S1*S1)/math.sqrt(math.pi*beta)
    cdef double B2 = 0.5*math.exp(-S2*S2)/math.sqrt(math.pi*beta)
    cdef double A1neg = 0.5*(1 - math.erf(S1))
    cdef double A2neg = 0.5*(1 - math.erf(S2))
    cdef double pr_by_rho = pr/rho
    cdef double u_sqr = ut*ut + un*un
    G[0] = rho*A2neg*(ut*A1neg - B1)

    cdef double temp1 = pr_by_rho + ut*ut
    cdef double temp2 = temp1*A1neg - ut*B1
    G[1] = rho*A2neg*temp2

    temp1 = ut*A1neg - B1
    temp2 = un*A2neg - B2
    G[2] = rho*temp1*temp2

    temp1 = (7*pr_by_rho) + u_sqr
    temp2 = 0.5*ut*temp1*A1neg
    
    temp1 = (6*pr_by_rho) + u_sqr
    cdef double temp3 = 0.5*B1*temp1 

    temp1 = ut*A1neg - B1
    cdef double temp4 = 0.5*rho*un*B2*temp1
        
    G[3] = rho*A2neg*(temp2 - temp3) - temp4
    return G

cdef np.ndarray[np.long] flux_quad_GxII(double nx, double ny, double u1, double u2, double rho, double pr):
    cdef np.ndarray G = np.zeros(4)
    cdef double tx = ny
    cdef double ty = -nx
    cdef double ut = u1*tx + u2*ty
    cdef double un = u1*nx + u2*ny

    cdef double beta = 0.5*rho/pr
    cdef double S1 = ut*math.sqrt(beta) 
    cdef double S2 = un*math.sqrt(beta) 
    cdef double B1 = 0.5*math.exp(-S1*S1)/math.sqrt(math.pi*beta)
    cdef double B2 = 0.5*math.exp(-S2*S2)/math.sqrt(math.pi*beta)
    cdef double A1pos = 0.5*(1 + math.erf(S1))     
    cdef double A2neg = 0.5*(1 - math.erf(S2))     

    cdef double pr_by_rho = pr/rho
    cdef double u_sqr = ut*ut + un*un

    G[0] = rho*A2neg*(ut*A1pos + B1)

    cdef double temp1 = pr_by_rho + ut*ut
    cdef double temp2 = temp1*A1pos + ut*B1
    G[1] = rho*A2neg*temp2

    temp1 = ut*A1pos + B1
    temp2 = un*A2neg - B2
    G[2] = rho*temp1*temp2

    temp1 = (7*pr_by_rho) + u_sqr
    temp2 = 0.5*ut*temp1*A1pos

    temp1 = (6*pr_by_rho) + u_sqr
    cdef double temp3 = 0.5*B1*temp1 

    temp1 = ut*A1pos + B1
    cdef double temp4 = 0.5*rho*un*B2*temp1

    G[3] = rho*A2neg*(temp2 + temp3) - temp4

    return G

cdef np.ndarray[np.long] flux_quad_GxIII(double nx, double ny, double u1, double u2, double rho, double pr):
    cdef np.ndarray G = np.zeros(4)
    cdef double tx = ny
    cdef double ty = -nx
    cdef double ut = u1*tx + u2*ty
    cdef double un = u1*nx + u2*ny

    cdef double beta = 0.5*rho/pr
    cdef double S1 = ut*math.sqrt(beta) 
    cdef double S2 = un*math.sqrt(beta) 
    cdef double B1 = 0.5*math.exp(-S1*S1)/math.sqrt(math.pi*beta)
    cdef double B2 = 0.5*math.exp(-S2*S2)/math.sqrt(math.pi*beta)
    cdef double A1pos = 0.5*(1 + math.erf(S1))     
    cdef double A2pos = 0.5*(1 + math.erf(S2))     

    cdef double pr_by_rho = pr/rho
    cdef double u_sqr = ut*ut + un*un


    G[0] = rho*A2pos*(ut*A1pos + B1)

    cdef double temp1 = pr_by_rho + ut*ut
    cdef double temp2 = temp1*A1pos + ut*B1
    G[1] = rho*A2pos*temp2

    temp1 = ut*A1pos + B1
    temp2 = un*A2pos + B2
    G[2] = rho*temp1*temp2

    temp1 = (7*pr_by_rho) + u_sqr
    temp2 = 0.5*ut*temp1*A1pos

    temp1 = (6*pr_by_rho) + u_sqr
    cdef double temp3 = 0.5*B1*temp1 

    temp1 = ut*A1pos + B1
    temp4 = 0.5*rho*un*B2*temp1

    G[3] = rho*A2pos*(temp2 + temp3) + temp4

    return G

cdef np.ndarray[np.long] flux_quad_GxIV(double nx, double ny, double u1, double u2, double rho, double pr):
    cdef np.ndarray G = np.zeros(4)
    cdef double tx = ny
    cdef double ty = -nx
    cdef double ut = u1*tx + u2*ty
    cdef double un = u1*nx + u2*ny

    cdef double beta = 0.5*rho/pr
    cdef double S1 = ut*math.sqrt(beta) 
    cdef double S2 = un*math.sqrt(beta) 
    cdef double B1 = 0.5*math.exp(-S1*S1)/math.sqrt(math.pi*beta)
    cdef double B2 = 0.5*math.exp(-S2*S2)/math.sqrt(math.pi*beta)
    cdef double A1neg = 0.5*(1 - math.erf(S1))     
    cdef double A2pos = 0.5*(1 + math.erf(S2))     

    cdef double pr_by_rho = pr/rho
    cdef double u_sqr = ut*ut + un*un

    G[0] = rho*A2pos*(ut*A1neg - B1)
        
    cdef double temp1 = pr_by_rho + ut*ut
    cdef double temp2 = temp1*A1neg - ut*B1
    G[1] = rho*A2pos*temp2

    temp1 = ut*A1neg - B1
    temp2 = un*A2pos + B2
    G[2] = rho*temp1*temp2

    temp1 = (7*pr_by_rho) + u_sqr
    temp2 = 0.5*ut*temp1*A1neg

    temp1 = (6*pr_by_rho) + u_sqr
    cdef double temp3 = 0.5*B1*temp1 

    temp1 = ut*A1neg - B1
    cdef double temp4 = 0.5*rho*un*B2*temp1

    G[3] = rho*A2pos*(temp2 - temp3) + temp4

    return G