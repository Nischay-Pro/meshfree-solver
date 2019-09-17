import math
import core
import numba
from numba import cuda
from cuda_func import multiply, multiply_element_wise, add, subtract, zeros, equalize

@cuda.jit(inline=True)
def func_delta_cuda_kernel(prim, prim_old, x, y, conn, nbhs, delta, cfl):
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x
    idx =  bx * bw + tx
    if idx > 0 and idx < len(prim):
        min_delt = 1
        equalize(prim_old[idx], prim[idx])
        x_i = x[idx]
        y_i = y[idx]
        for itm in conn[idx][:nbhs[idx]]:
            rho = prim[itm][0]
            u1 = prim[itm][1]
            u2 = prim[itm][2]
            pr = prim[itm][3]

            x_k = x[itm]
            y_k = y[itm]

            dist = (x_k - x_i)*(x_k - x_i) + (y_k - y_i)*(y_k - y_i)
            dist = math.sqrt(dist)

            mod_u = math.sqrt(u1*u1 + u2*u2)

            delta_t = dist/(mod_u + 3*math.sqrt(pr/rho))

            delta_t = cfl*delta_t

            if min_delt > delta_t:
                min_delt = delta_t

        delta[idx] = min_delt

@cuda.jit(inline=True)
def state_update_cuda(x, y, nx_gpu, ny_gpu, flag_1_gpu, nbhs, conn, prim, prim_old, delta, flux_res, Mach, gamma, pr_inf, rho_inf, aoa, sum_res_sqr_gpu, wall, interior, outer, rk, eu):
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x
    idx =  bx * bw + tx

    if idx > 0 and idx < len(x):

        U = cuda.local.array((4), numba.float64)
        U_old = cuda.local.array((4), numba.float64)
        temp1 = cuda.local.array((4), numba.float64)
        tempU = cuda.local.array((4), numba.float64)

        obt = 1 / 3
        tbt = 2 / 3

        zeros(U, U)
        zeros(U_old, U_old)
        zeros(temp1, temp1)
        zeros(tempU, tempU)

        sum_res_sqr_gpu[idx] = 0
        
        flag_1 = flag_1_gpu[idx]
        nx = nx_gpu[idx]
        ny = ny_gpu[idx]
        if flag_1 == wall:

            primitive_to_conserved_cuda_kernel(idx, nx, ny, U, prim[idx])
            primitive_to_conserved_cuda_kernel(idx, nx, ny, U_old, prim_old[idx])

            temp = U[0]

            if rk == 1 or rk == 2 or rk == 4:

                multiply(0.5 * eu * delta[idx], flux_res[idx], temp1)
                subtract(U, temp1, U)

            elif rk == 3:

                temp2 = cuda.local.array((4), numba.float64)
                zeros(temp2, temp2)
                
                multiply(0.5 * delta[idx], flux_res[idx], temp1)
                subtract(U, temp1, U)
                
                multiply(obt, U, U)

                multiply(tbt, U_old, U_old)

                add(U_old, U, U)

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

            prim[idx][0] = tempU[0]
            prim[idx][1] = tempU[1]
            prim[idx][2] = tempU[2]
            prim[idx][3] = tempU[3]

        elif flag_1 == outer:

            conserved_vector_Ubar_cuda_kernel(idx, nx, ny, Mach, gamma, pr_inf, rho_inf, aoa, U, prim[idx])
            conserved_vector_Ubar_cuda_kernel(idx, nx, ny, Mach, gamma, pr_inf, rho_inf, aoa, U_old, prim_old[idx])

            temp = U[0]

            if rk == 1 or rk == 2 or rk == 4:

                multiply(0.5 * eu * delta[idx], flux_res[idx], temp1)
                subtract(U, temp1, U)

            elif rk == 3:

                temp2 = cuda.local.array((4), numba.float64)
                zeros(temp2, temp2)
                
                multiply(0.5 * delta[idx], flux_res[idx], temp1)
                subtract(U, temp1, U)
                
                multiply(obt, U, U)

                multiply(tbt, U_old, U_old)

                add(U_old, U, U)

            U2_rot = U[1]
            U3_rot = U[2]

            U[1] = U2_rot*ny + U3_rot*nx
            U[2] = U3_rot*ny - U2_rot*nx

            tempU[0] = U[0]
            temp = 1 / U[0]
            tempU[1] = U[1]*temp
            tempU[2] = U[2]*temp
            tempU[3] = (0.4*U[3]) - (0.2*temp)*(U[1]*U[1] + U[2]*U[2])
            
            prim[idx][0] = tempU[0]
            prim[idx][1] = tempU[1]
            prim[idx][2] = tempU[2]
            prim[idx][3] = tempU[3]

        elif flag_1 == interior:

            primitive_to_conserved_cuda_kernel(idx, nx, ny, U, prim[idx])
            primitive_to_conserved_cuda_kernel(idx, nx, ny, U_old, prim_old[idx])

            temp = U[0]

            if rk == 1 or rk == 2 or rk == 4:

                multiply(0.5 * eu * delta[idx], flux_res[idx], temp1)
                subtract(U, temp1, U)

            elif rk == 3:

                temp2 = cuda.local.array((4), numba.float64)
                zeros(temp2, temp2)
                
                multiply(0.5 * delta[idx], flux_res[idx], temp1)
                subtract(U, temp1, U)
                
                multiply(obt, U, U)

                multiply(tbt, U_old, U_old)

                add(U_old, U, U)

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
            prim[idx][0] = tempU[0]
            prim[idx][1] = tempU[1]
            prim[idx][2] = tempU[2]
            prim[idx][3] = tempU[3]

@cuda.jit(device=True)
def primitive_to_conserved_cuda_kernel(itm, nx, ny, result, prim):

    U = cuda.local.array((4), numba.float64)

    rho = prim[0]
    U[0] = (rho) 
    temp1 = rho*prim[1]
    temp2 = rho*prim[2]

    U[1] = (temp1*ny - temp2*nx)
    U[2] = (temp1*nx + temp2*ny)
    U[3] = (2.5*prim[3] + 0.5*(temp1*temp1 + temp2*temp2)/rho)

    result[0] = U[0]
    result[1] = U[1]
    result[2] = U[2]
    result[3] = U[3]

@cuda.jit(device=True)
def conserved_vector_Ubar_cuda_kernel(itm, nx, ny, Mach, gamma, pr_inf, rho_inf, aoa, result, prim):

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

    rho = prim[0]
    u1 = prim[1]
    u2 = prim[2]
    pr = prim[3]

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