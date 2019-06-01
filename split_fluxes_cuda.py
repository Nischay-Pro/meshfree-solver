import math
from numba import cuda
import numba

@cuda.jit(device=True, inline=True)
def flux_Gxp(nx, ny, u1, u2, rho, pr, Gxp):

    tx = ny
    ty = -nx

    ut = u1*tx + u2*ty
    un = u1*nx + u2*ny

    beta = 0.5*rho/pr
    S1 = ut*math.sqrt(beta) 
    B1 = 0.5*math.exp(-S1*S1)/math.sqrt(math.pi*beta)
    A1pos = 0.5*(1 + math.erf(S1))     

    pr_by_rho = pr/rho
    u_sqr = ut*ut + un*un

    Gxp[0] = (rho*(ut*A1pos + B1))
        
    temp1 = pr_by_rho + ut*ut
    temp2 = temp1*A1pos + ut*B1
    Gxp[1] = (rho*temp2)

    temp1 = ut*un*A1pos + un*B1
    Gxp[2] = (rho*temp1)

    temp1 = (7*pr_by_rho) + u_sqr
    temp2 = 0.5*ut*temp1*A1pos 
    temp1 = (6*pr_by_rho) + u_sqr
    Gxp[3] = (rho*(temp2 + 0.5*temp1*B1))

@cuda.jit(device=True, inline=True)
def flux_Gxn(nx, ny, u1, u2, rho, pr, Gxn):

    tx = ny
    ty = -nx

    ut = u1*tx + u2*ty
    un = u1*nx + u2*ny

    beta = 0.5*rho/pr
    S1 = ut*math.sqrt(beta) 
    B1 = 0.5*math.exp(-S1*S1)/math.sqrt(math.pi*beta)
    A1neg = 0.5*(1 - math.erf(S1))     

    pr_by_rho = pr/rho
    u_sqr = ut*ut + un*un


    Gxn[0] = (rho*(ut*A1neg - B1))

    temp1 = pr_by_rho + ut*ut
    temp2 = temp1*A1neg - ut*B1
    Gxn[1] = (rho*temp2)

    temp1 = ut*un*A1neg - un*B1
    Gxn[2] = (rho*temp1)

    temp1 = (7*pr_by_rho) + u_sqr
    temp2 = 0.5*ut*temp1*A1neg 
    temp1 = (6*pr_by_rho) + u_sqr
    Gxn[3] = (rho*(temp2 - 0.5*temp1*B1))

@cuda.jit(device=True, inline=True)
def flux_Gyp(nx, ny, u1, u2, rho, pr, Gyp):

    tx = ny
    ty = -nx

    ut = u1*tx + u2*ty
    un = u1*nx + u2*ny

    beta = 0.5*rho/pr
    S2 = un*math.sqrt(beta) 
    B2 = 0.5*math.exp(-S2*S2)/math.sqrt(math.pi*beta)
    A2pos = 0.5*(1 + math.erf(S2))     

    pr_by_rho = pr/rho
    u_sqr = ut*ut + un*un


    Gyp[0] = (rho*(un*A2pos + B2))

    temp1 = pr_by_rho + un*un
    temp2 = temp1*A2pos + un*B2

    temp1 = ut*un*A2pos + ut*B2
    Gyp[1] = (rho*temp1)

    Gyp[2] = (rho*temp2)

    temp1 = (7*pr_by_rho) + u_sqr
    temp2 = 0.5*un*temp1*A2pos 
    temp1 = (6*pr_by_rho) + u_sqr
    Gyp[3] = (rho*(temp2 + 0.5*temp1*B2))

@cuda.jit(device=True, inline=True)
def flux_Gyn(nx, ny, u1, u2, rho, pr, Gyn):

    tx = ny
    ty = -nx

    ut = u1*tx + u2*ty
    un = u1*nx + u2*ny

    beta = 0.5*rho/pr
    S2 = un*math.sqrt(beta) 
    B2 = 0.5*math.exp(-S2*S2)/math.sqrt(math.pi*beta)
    A2neg = 0.5*(1 - math.erf(S2))     

    pr_by_rho = pr/rho
    u_sqr = ut*ut + un*un

    Gyn[0] = (rho*(un*A2neg - B2))
    
    temp1 = pr_by_rho + un*un
    temp2 = temp1*A2neg - un*B2

    temp1 = ut*un*A2neg - ut*B2
    Gyn[1] = (rho*temp1)

    Gyn[2] = (rho*temp2)

    temp1 = (7*pr_by_rho) + u_sqr
    temp2 = 0.5*un*temp1*A2neg 
    temp1 = (6*pr_by_rho) + u_sqr
    Gyn[3] = (rho*(temp2 - 0.5*temp1*B2))

# @cuda.jit(device=True, inline=True)
# def flux_Gx(Gx, nx, ny, u1, u2, rho, pr, Gx):
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