import math
from numba import cuda
import numba

@cuda.jit(device=True)
def flux_Gxp(nx, ny, shared, op):

    u1 = shared[cuda.threadIdx.x + cuda.blockDim.x * 4]
    u2 = shared[cuda.threadIdx.x + cuda.blockDim.x * 5]
    rho = shared[cuda.threadIdx.x + cuda.blockDim.x * 6]
    pr = shared[cuda.threadIdx.x + cuda.blockDim.x * 7]

    ut = u1*ny - u2*nx
    un = u1*nx + u2*ny

    beta = 0.5*rho/pr
    S1 = ut*math.sqrt(beta) 
    B1 = 0.5*math.exp(-S1*S1)/math.sqrt(math.pi*beta)
    A1pos = 0.5*(1 + math.erf(S1))     

    pr_by_rho = pr/rho
    u_sqr = ut*ut + un*un

    shared[cuda.threadIdx.x] = op((rho*(ut*A1pos + B1)), shared[cuda.threadIdx.x])
        
    temp1 = pr_by_rho + ut*ut
    temp2 = temp1*A1pos + ut*B1
    shared[cuda.threadIdx.x + cuda.blockDim.x] = op((rho*temp2), shared[cuda.threadIdx.x + cuda.blockDim.x])

    temp1 = ut*un*A1pos + un*B1
    shared[cuda.threadIdx.x + cuda.blockDim.x * 2] = op((rho*temp1), shared[cuda.threadIdx.x + cuda.blockDim.x * 2])

    temp1 = (7*pr_by_rho) + u_sqr
    temp2 = 0.5*ut*temp1*A1pos 
    temp1 = (6*pr_by_rho) + u_sqr
    shared[cuda.threadIdx.x + cuda.blockDim.x * 3] = op((rho*(temp2 + 0.5*temp1*B1)), shared[cuda.threadIdx.x + cuda.blockDim.x * 3])

@cuda.jit(device=True)
def flux_Gxn(nx, ny, shared, op):

    u1 = shared[cuda.threadIdx.x + cuda.blockDim.x * 4]
    u2 = shared[cuda.threadIdx.x + cuda.blockDim.x * 5]
    rho = shared[cuda.threadIdx.x + cuda.blockDim.x * 6]
    pr = shared[cuda.threadIdx.x + cuda.blockDim.x * 7]

    ut = u1*ny - u2*nx
    un = u1*nx + u2*ny

    beta = 0.5*rho/pr
    S1 = ut*math.sqrt(beta) 
    B1 = 0.5*math.exp(-S1*S1)/math.sqrt(math.pi*beta)
    A1neg = 0.5*(1 - math.erf(S1))     

    pr_by_rho = pr/rho
    u_sqr = ut*ut + un*un


    shared[cuda.threadIdx.x] = op((rho*(ut*A1neg - B1)), shared[cuda.threadIdx.x])

    temp1 = pr_by_rho + ut*ut
    temp2 = temp1*A1neg - ut*B1
    shared[cuda.threadIdx.x + cuda.blockDim.x] = op((rho*temp2), shared[cuda.threadIdx.x + cuda.blockDim.x])

    temp1 = ut*un*A1neg - un*B1
    shared[cuda.threadIdx.x + cuda.blockDim.x * 2] = op((rho*temp1), shared[cuda.threadIdx.x + cuda.blockDim.x * 2])

    temp1 = (7*pr_by_rho) + u_sqr
    temp2 = 0.5*ut*temp1*A1neg 
    temp1 = (6*pr_by_rho) + u_sqr
    shared[cuda.threadIdx.x + cuda.blockDim.x * 3] = op((rho*(temp2 - 0.5*temp1*B1)), shared[cuda.threadIdx.x + cuda.blockDim.x * 3])

@cuda.jit(device=True)
def flux_Gyp(nx, ny, shared, op):

    u1 = shared[cuda.threadIdx.x + cuda.blockDim.x * 4]
    u2 = shared[cuda.threadIdx.x + cuda.blockDim.x * 5]
    rho = shared[cuda.threadIdx.x + cuda.blockDim.x * 6]
    pr = shared[cuda.threadIdx.x + cuda.blockDim.x * 7]

    ut = u1*ny - u2*nx
    un = u1*nx + u2*ny

    beta = 0.5*rho/pr
    S2 = un*math.sqrt(beta) 
    B2 = 0.5*math.exp(-S2*S2)/math.sqrt(math.pi*beta)
    A2pos = 0.5*(1 + math.erf(S2))     

    pr_by_rho = pr/rho
    u_sqr = ut*ut + un*un


    shared[cuda.threadIdx.x] = op((rho*(un*A2pos + B2)), shared[cuda.threadIdx.x])

    temp1 = pr_by_rho + un*un
    temp2 = temp1*A2pos + un*B2

    temp1 = ut*un*A2pos + ut*B2
    shared[cuda.threadIdx.x + cuda.blockDim.x] = op((rho*temp1), shared[cuda.threadIdx.x + cuda.blockDim.x])

    shared[cuda.threadIdx.x + cuda.blockDim.x * 2] = op((rho*temp2), shared[cuda.threadIdx.x + cuda.blockDim.x * 2])

    temp1 = (7*pr_by_rho) + u_sqr
    temp2 = 0.5*un*temp1*A2pos 
    temp1 = (6*pr_by_rho) + u_sqr
    shared[cuda.threadIdx.x + cuda.blockDim.x * 3] = op((rho*(temp2 + 0.5*temp1*B2)), shared[cuda.threadIdx.x + cuda.blockDim.x * 3])

@cuda.jit(device=True)
def flux_Gyn(nx, ny, shared, op):

    u1 = shared[cuda.threadIdx.x + cuda.blockDim.x * 4]
    u2 = shared[cuda.threadIdx.x + cuda.blockDim.x * 5]
    rho = shared[cuda.threadIdx.x + cuda.blockDim.x * 6]
    pr = shared[cuda.threadIdx.x + cuda.blockDim.x * 7]
    
    ut = u1*ny - u2*nx
    un = u1*nx + u2*ny

    beta = 0.5*rho/pr
    S2 = un*math.sqrt(beta) 
    B2 = 0.5*math.exp(-S2*S2)/math.sqrt(math.pi*beta)
    A2neg = 0.5*(1 - math.erf(S2))     

    pr_by_rho = pr/rho
    u_sqr = ut*ut + un*un

    shared[cuda.threadIdx.x] = op((rho*(un*A2neg - B2)), shared[cuda.threadIdx.x])
    
    temp1 = pr_by_rho + un*un
    temp2 = temp1*A2neg - un*B2
 
    temp1 = ut*un*A2neg - ut*B2
    shared[cuda.threadIdx.x + cuda.blockDim.x] = op((rho*temp1), shared[cuda.threadIdx.x + cuda.blockDim.x])

    shared[cuda.threadIdx.x + cuda.blockDim.x * 2] = op((rho*temp2), shared[cuda.threadIdx.x + cuda.blockDim.x * 2])

    temp1 = (7*pr_by_rho) + u_sqr
    temp2 = 0.5*un*temp1*A2neg 
    temp1 = (6*pr_by_rho) + u_sqr
    shared[cuda.threadIdx.x + cuda.blockDim.x * 3] = op((rho*(temp2 - 0.5*temp1*B2)), shared[cuda.threadIdx.x + cuda.blockDim.x * 3])