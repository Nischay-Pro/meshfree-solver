@inline function func_delta_kernel(gpuGlobalDataConn,  gpuGlobalDataFixedPoint, gpuGlobalDataRest, gpuConfigData)
    cfl = gpuConfigData[2]
    tx = threadIdx().x
    bx = blockIdx().x - 1
    bw = blockDim().x
    idx = bx * bw + tx
    if idx > 0 && idx <= gpuGlobalDataFixedPoint[end].localID
        # TODO - Possible problem?
        min_delt = one(Float64)
        for iter in 5:14
            conn = gpuGlobalDataConn[idx, iter]
            if conn == zero(Float64)
                break
            end
            rho = gpuGlobalDataRest[1, conn]
            u1 = gpuGlobalDataRest[2, conn]
            u2 = gpuGlobalDataRest[3, conn]
            pr = gpuGlobalDataRest[4, conn]
            x_i = gpuGlobalDataFixedPoint[idx].x
            y_i = gpuGlobalDataFixedPoint[idx].y
            x_k = gpuGlobalDataFixedPoint[conn].x
            y_k = gpuGlobalDataFixedPoint[conn].y
            dist = CUDAnative.hypot((x_k - x_i),(y_k - y_i))
            mod_u = CUDAnative.hypot(u1,u2)
            delta_t = dist/(mod_u + 3*CUDAnative.sqrt(pr/rho))
            delta_t *= cfl
            if min_delt > delta_t
                min_delt = delta_t
            end
        end
        gpuGlobalDataRest[29, idx] = min_delt
        gpuGlobalDataRest[30, idx] = gpuGlobalDataRest[1, idx]
        gpuGlobalDataRest[31, idx] = gpuGlobalDataRest[2, idx]
        gpuGlobalDataRest[32, idx] = gpuGlobalDataRest[3, idx]
        gpuGlobalDataRest[33, idx] = gpuGlobalDataRest[4, idx]
    end
    # sync_threads()
    return nothing
end

@inline function state_update_kernel(gpuGlobalDataFixedPoint, gpuGlobalDataConn, gpuGlobalDataRest,  gpuConfigData, gpuSumResSqr, numPoints, rk)
    tx = threadIdx().x
    bx = blockIdx().x - 1
    bw = blockDim().x
	idx = bx * bw + tx
    max_res = zero(Float64)
    if idx > 0 && idx <= numPoints
        min_delt = gpuGlobalDataRest[29, idx]
        # min_delt = min_delt
        flag1 = gpuGlobalDataFixedPoint[idx].flag_1
        if flag1 == gpuConfigData[17]
            state_update_wall_kernel(gpuGlobalDataFixedPoint, gpuGlobalDataRest, idx, min_delt, rk)
        end
        if flag1 == gpuConfigData[18]
            state_update_interior_kernel(gpuGlobalDataFixedPoint, gpuGlobalDataRest, idx, min_delt, rk)
        end
        if flag1 == gpuConfigData[19]
            state_update_outer_kernel(gpuGlobalDataFixedPoint, gpuGlobalDataRest, gpuConfigData, idx, min_delt, rk)
        end
        if flag1 != gpuConfigData[19]
            gpuSumResSqr[idx] = min_delt * gpuGlobalDataRest[5, idx] * min_delt * gpuGlobalDataRest[5, idx]
        end
    end
    # sync_threads()
    return nothing
end

function state_update_wall_kernel(gpuGlobalDataFixedPoint, gpuGlobalDataRest, idx, min_delt, rk)
    nx = gpuGlobalDataFixedPoint[idx].nx
    ny = gpuGlobalDataFixedPoint[idx].ny

    rho = gpuGlobalDataRest[1, idx]
    U1 = rho
    temp1 = rho * gpuGlobalDataRest[2, idx]
    temp2 = rho * gpuGlobalDataRest[3, idx]
    U2 = temp1*ny - temp2*nx
    U3 = temp1*nx + temp2*ny
    U4 = 2.5*gpuGlobalDataRest[4, idx] + 0.5*(temp1*temp1 + temp2*temp2)/rho

    # if idx == 3
    #     @cuprintf("\n ======= ")
    #     @cuprintf("\n Values are %f %f %f %f \n", U1,U2, U3, U4)
    # end

    temp = U1
    U1 -= 0.5 * min_delt * gpuGlobalDataRest[5, idx]
    U2 -= 0.5 * min_delt * gpuGlobalDataRest[6, idx]
    U3 -= 0.5 * min_delt * gpuGlobalDataRest[7, idx]
    U4 -= 0.5 * min_delt * gpuGlobalDataRest[8, idx]
    if rk == 3
        rho = gpuGlobalDataRest[30, idx]
        U1_old = rho
        temp1 = rho * gpuGlobalDataRest[31, idx]
        temp2 = rho * gpuGlobalDataRest[32, idx]
        U2_old = temp1*ny - temp2*nx
        U3_old = temp1*nx + temp2*ny
        U4_old = 2.5*gpuGlobalDataRest[33, idx] + 0.5*(temp1*temp1 + temp2*temp2)/rho
        U1 = U1 * 1/3 + 2/3 * U1_old
        U2 = U2 * 1/3 + 2/3 * U2_old
        U3 = U3 * 1/3 + 2/3 * U3_old
        U4 = U4 * 1/3 + 2/3 * U4_old
    end
    # if idx == 3
    #     @cuprintf("\n ======= ")
    #     @cuprintf("\n Values are %f %f %f %f \n", U1,U2, U3, U4)
    # end
    U3 = zero(Float64)
    U2_rot = U2
    U3_rot = U3
    U2 = U2_rot*ny + U3_rot*nx
    U3 = U3_rot*ny - U2_rot*nx
    # res_sqr = (U1 - temp)*(U1 - temp)
    # gpuSumResSqr[idx] = res_sqr
    # if idx == 3
    #     @cuprintf("\n ======= ")
    #     @cuprintf("\n Values are %f %f %f %f \n", U1,U2, U3, U4)
    # end

    # if res_sqr > max_res
    #     max_res = res_sqr
    #     max_res_point = idx
    # end
    # sum_res_sqr = sum_res_sqr + res_sqr

    gpuGlobalDataRest[1, idx] = U1
    temp = 1.0 / U1
    gpuGlobalDataRest[2, idx] = U2*temp
    gpuGlobalDataRest[3, idx] = U3*temp
    gpuGlobalDataRest[4, idx] = (0.4*U4) - ((0.2 * temp) * (U2 * U2 + U3 * U3))
    return nothing
end

function state_update_interior_kernel(gpuGlobalDataFixedPoint, gpuGlobalDataRest, idx, min_delt, rk)
    nx = gpuGlobalDataFixedPoint[idx].nx
    ny = gpuGlobalDataFixedPoint[idx].ny

    rho = gpuGlobalDataRest[1, idx]
    U1 = rho
    temp1 = rho * gpuGlobalDataRest[2, idx]
    temp2 = rho * gpuGlobalDataRest[3, idx]
    U2 = temp1*ny - temp2*nx
    U3 = temp1*nx + temp2*ny
    U4 = 2.5*gpuGlobalDataRest[4, idx] + 0.5*(temp1*temp1 + temp2*temp2)/rho
    # if idx == 1
    #     @cuprintf("\n %.17f %.17f %.17f %.17f", U1, U2, U3, U4)
    #     # @cuprintf("\n %.17f ", temp1)
    # end
    # res_sqr = (U1 - temp)*(U1 - temp)
    # gpuSumResSqr[idx] = res_sqr
    temp = U1
    U1 -= 0.5 * min_delt * gpuGlobalDataRest[5, idx]
    U2 -= 0.5 * min_delt * gpuGlobalDataRest[6, idx]
    U3 -= 0.5 * min_delt * gpuGlobalDataRest[7, idx]
    U4 -= 0.5 * min_delt * gpuGlobalDataRest[8, idx]
    if rk == 3
        rho = gpuGlobalDataRest[30, idx]
        U1_old = rho
        temp1 = rho * gpuGlobalDataRest[31, idx]
        temp2 = rho * gpuGlobalDataRest[32, idx]
        U2_old = temp1*ny - temp2*nx
        U3_old = temp1*nx + temp2*ny
        U4_old = 2.5*gpuGlobalDataRest[33, idx] + 0.5*(temp1*temp1 + temp2*temp2)/rho
        U1 = U1 * 1/3 + 2/3 * U1_old
        U2 = U2 * 1/3 + 2/3 * U2_old
        U3 = U3 * 1/3 + 2/3 * U3_old
        U4 = U4 * 1/3 + 2/3 * U4_old
    end
    U2_rot = U2
    U3_rot = U3
    U2 = U2_rot*ny + U3_rot*nx
    U3 = U3_rot*ny - U2_rot*nx
    # if idx == 1
    #     @cuprintf("\n %.17f %.17f %.17f %.17f", U1, U2, U3, U4)
    #     @cuprintf("\n %.17f ", temp)
    # end
    gpuGlobalDataRest[1, idx] = U1
    temp = 1.0 / U1
    gpuGlobalDataRest[2, idx] = U2*temp
    gpuGlobalDataRest[3, idx] = U3*temp
    gpuGlobalDataRest[4, idx] = (0.4*U4) - ((0.2 * temp) * (U2 * U2 + U3 * U3))
    return nothing
end

function state_update_outer_kernel(gpuGlobalDataFixedPoint, gpuGlobalDataRest, gpuConfigData, idx, min_delt, rk)
    nx = gpuGlobalDataFixedPoint[idx].nx
    ny = gpuGlobalDataFixedPoint[idx].ny
    Mach::Float64 = gpuConfigData[4]
    rho_inf::Float64 = gpuConfigData[12]
    pr_inf::Float64 = gpuConfigData[13]
    gamma::Float64 = gpuConfigData[15]
    theta = deg2rad(gpuConfigData[5])

    u1_inf::Float64 = Mach*CUDAnative.cos(theta)
    u2_inf::Float64 = Mach*CUDAnative.sin(theta)

    tx = ny
    ty = -nx

    u1_inf_rot = u1_inf*tx + u2_inf*ty
    u2_inf_rot = u1_inf*nx + u2_inf*ny

    temp1 = (u1_inf_rot * u1_inf_rot + u2_inf_rot*u2_inf_rot)
    e_inf = (pr_inf/(rho_inf*(gamma-1))) + 0.5 * (temp1)

    beta = (0.5 * rho_inf)/pr_inf
    S2 = u2_inf_rot * CUDAnative.sqrt(beta)
    B2_inf = CUDAnative.exp(-S2*S2)/(2*CUDAnative.sqrt(pi*beta))
    A2n_inf = 0.5 * (1 - CUDAnative.erf(S2))

    rho = gpuGlobalDataRest[1, idx]
    u1 = gpuGlobalDataRest[2, idx]
    u2 = gpuGlobalDataRest[3, idx]
    pr = gpuGlobalDataRest[4, idx]

    u1_rot = u1*tx + u2*ty
    u2_rot = u1*nx + u2*ny

    temp1 = (u1_rot*u1_rot + u2_rot*u2_rot)
    e = (pr/(rho*(gamma-1))) + 0.5*(temp1)

    beta = (rho)/(2*pr)
    S2 = u2_rot*CUDAnative.sqrt(beta)
    B2 = CUDAnative.exp(-S2*S2)/(2*CUDAnative.sqrt(pi*beta))
    A2p = 0.5*(1 + CUDAnative.erf(S2))

    U1 = (rho_inf*A2n_inf) + (rho*A2p)

    U2 = (rho_inf*u1_inf_rot*A2n_inf) + (rho*u1_rot*A2p)

    temp1 = rho_inf*(u2_inf_rot*A2n_inf - B2_inf)
    temp2 = rho*(u2_rot*A2p + B2)
    U3 = (temp1 + temp2)

    temp1 = (rho_inf*A2n_inf* e_inf - 0.5*rho_inf*u2_inf_rot*B2_inf)
    temp2 = (rho*A2p*e + 0.5*rho*u2_rot*B2)

    U4 = (temp1 + temp2)

    temp = U1
    U1 -= 0.5 * min_delt * gpuGlobalDataRest[5, idx]
    U2 -= 0.5 * min_delt * gpuGlobalDataRest[6, idx]
    U3 -= 0.5 * min_delt * gpuGlobalDataRest[7, idx]
    U4 -= 0.5 * min_delt * gpuGlobalDataRest[8, idx]
    if rk == 3
        rho = gpuGlobalDataRest[30, idx]
        u1 = gpuGlobalDataRest[31, idx]
        u2 = gpuGlobalDataRest[32, idx]
        pr = gpuGlobalDataRest[33, idx]

        u1_rot = u1*tx + u2*ty
        u2_rot = u1*nx + u2*ny

        temp1 = (u1_rot*u1_rot + u2_rot*u2_rot)
        e = (pr/(rho*(gamma-1))) + 0.5*(temp1)

        beta = (rho)/(2*pr)
        S2 = u2_rot*CUDAnative.sqrt(beta)
        B2 = CUDAnative.exp(-S2*S2)/(2*CUDAnative.sqrt(pi*beta))
        A2p = 0.5*(1 + CUDAnative.erf(S2))

        U1_old = (rho_inf*A2n_inf) + (rho*A2p)

        U2_old = (rho_inf*u1_inf_rot*A2n_inf) + (rho*u1_rot*A2p)

        temp1 = rho_inf*(u2_inf_rot*A2n_inf - B2_inf)
        temp2 = rho*(u2_rot*A2p + B2)
        U3_old = (temp1 + temp2)

        temp1 = (rho_inf*A2n_inf* e_inf - 0.5*rho_inf*u2_inf_rot*B2_inf)
        temp2 = (rho*A2p*e + 0.5*rho*u2_rot*B2)

        U4_old = (temp1 + temp2)

        U1 = U1 * 1/3 + 2/3 * U1_old
        U2 = U2 * 1/3 + 2/3 * U2_old
        U3 = U3 * 1/3 + 2/3 * U3_old
        U4 = U4 * 1/3 + 2/3 * U4_old
    end

    U2_rot = U2
    U3_rot = U3
    U2 = U2_rot*ny + U3_rot*nx
    U3 = U3_rot*ny - U2_rot*nx
    gpuGlobalDataRest[1, idx] = U1
    temp = 1.0 / U1
    gpuGlobalDataRest[2, idx] = U2*temp
    gpuGlobalDataRest[3, idx] = U3*temp
    gpuGlobalDataRest[4, idx] = (0.4*U4) - ((0.2 * temp) * (U2 * U2 + U3 * U3))
    return nothing
end
