import math
import core
import numba
from numba import cuda
from cuda_func import multiply, multiply_element_wise, add, subtract, zeros

@cuda.jit(inline=True)
def func_delta_cuda_kernel(globaldata, cfl):
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x
    idx =  bx * bw + tx
    if idx > 0 and idx < len(globaldata):
        min_delt = 1
        x_i = globaldata[idx]['x']
        y_i = globaldata[idx]['y']
        for itm in globaldata[idx]['conn'][:globaldata[idx]['nbhs']]:
            rho = globaldata[itm]['prim'][0]
            u1 = globaldata[itm]['prim'][1]
            u2 = globaldata[itm]['prim'][2]
            pr = globaldata[itm]['prim'][3]

            x_k = globaldata[itm]['x']
            y_k = globaldata[itm]['y']

            dist = (x_k - x_i)*(x_k - x_i) + (y_k - y_i)*(y_k - y_i)
            dist = math.sqrt(dist)

            mod_u = math.sqrt(u1*u1 + u2*u2)

            delta_t = dist/(mod_u + 3*math.sqrt(pr/rho))

            delta_t = cfl*delta_t

            if min_delt > delta_t:
                min_delt = delta_t

        globaldata[idx]['delta'] = min_delt

@cuda.jit(inline=True)
def state_update_cuda(globaldata, Mach, gamma, pr_inf, rho_inf, aoa, sum_res_sqr_gpu, wall, interior, outer):
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x
    idx =  bx * bw + tx
    sum_res_sqr_gpu[idx] = 0

    U = cuda.local.array((4), numba.float64)
    temp1 = cuda.local.array((4), numba.float64)
    tempU = cuda.local.array((4), numba.float64)

    zeros(U, U)
    zeros(temp1, temp1)
    zeros(tempU, tempU)

    if idx > 0 and idx < len(globaldata):
        flag_1 = globaldata[idx]['flag_1']
        nx = globaldata[idx]['nx']
        ny = globaldata[idx]['ny']
        if flag_1 == wall:

            primitive_to_conserved_cuda_kernel(globaldata, idx, nx, ny, U)

            temp = U[0]

            multiply(globaldata[idx]['delta'], globaldata[idx]['flux_res'], temp1)
            subtract(U, temp1, U)

            U[2] = 0

            U2_rot = U[1]
            U3_rot = U[2]
            U[1] = U2_rot*ny + U3_rot*nx
            U[2] = U3_rot*ny - U2_rot*nx

            res_sqr = (U[0] - temp) ** 2
            sum_res_sqr_gpu[idx] = res_sqr

            tempU[0] = U[0]
            temp = 1 / U[0]
            tempU[1] = U[1]*temp
            tempU[2] = U[2]*temp

            tempU[3] = (0.4*U[3]) - ((0.2*temp)*(U[1]*U[1] + U[2]*U[2]))

            globaldata[idx]['prim'][0] = tempU[0]
            globaldata[idx]['prim'][1] = tempU[1]
            globaldata[idx]['prim'][2] = tempU[2]
            globaldata[idx]['prim'][3] = tempU[3]

        elif flag_1 == outer:

            conserved_vector_Ubar_cuda_kernel(globaldata, idx, nx, ny, Mach, gamma, pr_inf, rho_inf, aoa, U)

            temp = U[0]

            multiply(globaldata[idx]['delta'], globaldata[idx]['flux_res'], temp1)
            subtract(U, temp1, U)

            U2_rot = U[1]
            U3_rot = U[2]

            U[1] = U2_rot*ny + U3_rot*nx
            U[2] = U3_rot*ny - U2_rot*nx

            tempU[0] = U[0]
            temp = 1 / U[0]
            tempU[1] = U[1]*temp
            tempU[2] = U[2]*temp
            tempU[3] = (0.4*U[3]) - (0.2*temp)*(U[1]*U[1] + U[2]*U[2])
            
            globaldata[idx]['prim'][0] = tempU[0]
            globaldata[idx]['prim'][1] = tempU[1]
            globaldata[idx]['prim'][2] = tempU[2]
            globaldata[idx]['prim'][3] = tempU[3]

        elif flag_1 == interior:

            primitive_to_conserved_cuda_kernel(globaldata, idx, nx, ny, U)

            temp = U[0]

            multiply(globaldata[idx]['delta'], globaldata[idx]['flux_res'], temp1)
            subtract(U, temp1, U)

            U2_rot = U[1]
            U3_rot = U[2]
            U[1] = U2_rot*ny + U3_rot*nx
            U[2] = U3_rot*ny - U2_rot*nx

            res_sqr = (U[0] - temp) ** 2
            sum_res_sqr_gpu[idx] = res_sqr

            tempU[0] = U[0]
            temp = 1 / U[0]
            tempU[1] = U[1]*temp
            tempU[2] = U[2]*temp
            tempU[3] = (0.4*U[3]) - (0.2*temp)*(U[1]*U[1] + U[2]*U[2])
            globaldata[idx]['prim'][0] = tempU[0]
            globaldata[idx]['prim'][1] = tempU[1]
            globaldata[idx]['prim'][2] = tempU[2]
            globaldata[idx]['prim'][3] = tempU[3]

@cuda.jit(device=True, inline=True)
def primitive_to_conserved_cuda_kernel(globaldata, itm, nx, ny, result):

    U = cuda.local.array((4), numba.float64)

    rho = globaldata[itm]['prim'][0]
    U[0] = (rho) 
    temp1 = rho*globaldata[itm]['prim'][1]
    temp2 = rho*globaldata[itm]['prim'][2]

    U[1] = (temp1*ny - temp2*nx)
    U[2] = (temp1*nx + temp2*ny)
    U[3] = (2.5*globaldata[itm]['prim'][3] + 0.5*(temp1*temp1 + temp2*temp2)/rho)

    result[0] = U[0]
    result[1] = U[1]
    result[2] = U[2]
    result[3] = U[3]

@cuda.jit(device=True, inline=True)
def conserved_vector_Ubar_cuda_kernel(globaldata, itm, nx, ny, Mach, gamma, pr_inf, rho_inf, aoa, result):

    Ubar = cuda.local.array((4), numba.float64)

    theta = cuda.local.array((1), numba.float64)

    core.calculateThetaCuda(aoa, theta)

    u1_inf = Mach*math.cos(theta[0])
    u2_inf = Mach*math.sin(theta[0])

    tx = ny
    ty = -nx

    u1_inf_rot = u1_inf*tx + u2_inf*ty
    u2_inf_rot = u1_inf*nx + u2_inf*ny

    temp1 = (u1_inf_rot*u1_inf_rot + u2_inf_rot*u2_inf_rot)
    e_inf = pr_inf/(rho_inf*(gamma-1)) + 0.5*(temp1)

    beta = (0.5*rho_inf)/pr_inf
    S2 = u2_inf_rot*math.sqrt(beta)
    B2_inf = math.exp(-S2*S2)/(2*math.sqrt(math.pi*beta))
    A2n_inf = 0.5*(1-math.erf(S2))

    rho = globaldata[itm]['prim'][0]
    u1 = globaldata[itm]['prim'][1]
    u2 = globaldata[itm]['prim'][2]
    pr = globaldata[itm]['prim'][3]

    u1_rot = u1*tx + u2*ty
    u2_rot = u1*nx + u2*ny

    temp1 = (u1_rot*u1_rot + u2_rot*u2_rot)
    e = pr/(rho*(gamma-1)) + 0.5*(temp1)

    beta = (rho)/(2*pr)
    S2 = u2_rot*math.sqrt(beta)
    B2 = math.exp(-S2*S2)/(2*math.sqrt(math.pi*beta))
    A2p = 0.5*(1+math.erf(S2))

    Ubar[0] = ((rho_inf*A2n_inf) + (rho*A2p))

    Ubar[1] = ((rho_inf*u1_inf_rot*A2n_inf) + (rho*u1_rot*A2p))

    temp1 = rho_inf*(u2_inf_rot*A2n_inf - B2_inf)
    temp2 = rho*(u2_rot*A2p + B2)
    Ubar[2] = (temp1 + temp2)

    temp1 = (rho_inf*A2n_inf*e_inf - 0.5*rho_inf*u2_inf_rot*B2_inf)
    temp2 = (rho*A2p*e + 0.5*rho*u2_rot*B2)

    Ubar[3] = (temp1 + temp2)

    result[0] = Ubar[0]
    result[1] = Ubar[1]
    result[2] = Ubar[2]
    result[3] = Ubar[3]