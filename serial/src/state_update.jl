function func_delta(globaldata, numPoints, cfl)
    for idx in 1:numPoints
        min_delt = one(Float64)
        for conn in globaldata.conn[idx]
            if conn == 0
                break
            end
            x_i = globaldata.x[idx]
            y_i = globaldata.y[idx]
            x_k = globaldata.x[conn]
            y_k = globaldata.y[conn]
            dist = hypot((x_k - x_i),(y_k - y_i))
            mod_u = hypot(globaldata.prim[conn][2],globaldata.prim[conn][3])
            delta_t = dist/(mod_u + 3*sqrt(globaldata.prim[conn][4]/globaldata.prim[conn][1]))
            delta_t *= cfl
            if min_delt > delta_t
                min_delt = delta_t
            end
        end
        globaldata.delta[idx] = min_delt
        @. globaldata.prim_old[idx] = globaldata.prim[idx]
    end
    return nothing
end

function state_update(globaldata, numPoints, configData, iter, res_old, rk, U, Uold, main_store)
    max_res = zero(Float64)
    ∑_res_sqr = zeros(Float64, 1)
    Mach = main_store[58] 
    gamma = main_store[59] 
    pr_inf = main_store[60] 
    rho_inf = main_store[61] 
    theta = main_store[62]

    for idx in 1:numPoints
        if globaldata.flag_1[idx] == 0
            fill!(U, zero(Float64))
            state_update_wall(globaldata, idx, max_res, ∑_res_sqr, U, Uold, rk)
        elseif globaldata.flag_1[idx] == 2
            fill!(U, zero(Float64))
            state_update_outer(globaldata, idx, Mach, gamma, pr_inf, rho_inf, theta, max_res, ∑_res_sqr, U, Uold, rk)
        elseif globaldata.flag_1[idx] == 1
            fill!(U, zero(Float64))
            state_update_interior(globaldata, idx, max_res, ∑_res_sqr, U, Uold, rk)
        end
    end

    res_new = sqrt(∑_res_sqr[1])/ numPoints
    residue = zero(Float64)

    if iter <= 2
        res_old[1] = res_new
        residue = zero(Float64)
    else
        residue = log10(res_new/res_old[1])
    end
    # println(residue)
    if rk == 4
        @printf("%.17f \n", residue)
    end
    # open("residue_" * string(numPoints) * ".txt", "a+") do residue_io
    #     @printf(residue_io, "%d %s\n", iter, residue)
    # end

    # res_old = 0
    return nothing
end

function state_update_wall(globaldata, idx, max_res, ∑_res_sqr, U, Uold, rk)
    nx = globaldata.nx[idx]
    ny = globaldata.ny[idx]

    primitive_to_conserved(globaldata.prim[idx], nx, ny, U)
    primitive_to_conserved_old(globaldata.prim_old[idx], nx, ny, Uold)

    temp = U[1]
    for iter in 1:4
        U[iter] = U[iter] - 0.5 * globaldata.delta[idx][iter] * globaldata.flux_res[idx][iter]
    end
    if rk == 3
        for iter in 1:4
            U[iter] = U[iter] * 1/3 + Uold[iter] * 2/3
        end
    end
    U[3] = zero(Float64)
    U2_rot = U[2]
    U3_rot = U[3]
    U[2] = U2_rot*ny + U3_rot*nx
    U[3] = U3_rot*ny - U2_rot*nx
    res_sqr = (U[1] - temp)*(U[1] - temp)

    ∑_res_sqr[1] += res_sqr
    globaldata.prim[idx][1] = U[1]
    temp = 1.0 / U[1]
    globaldata.prim[idx][2] = U[2]*temp
    globaldata.prim[idx][3] = U[3]*temp
    globaldata.prim[idx][4] = (0.4*U[4]) - ((0.2 * temp) * (U[2] * U[2] + U[3] * U[3]))
    return nothing
end

function state_update_outer(globaldata, idx, Mach, gamma, pr_inf, rho_inf, theta, max_res, ∑_res_sqr, U, Uold, rk)
    nx = globaldata.nx[idx]
    ny = globaldata.ny[idx]
    conserved_vector_Ubar(globaldata.prim[idx], nx, ny, Mach, gamma, pr_inf, rho_inf, theta, U)
    conserved_vector_Ubar_old(globaldata.prim_old[idx], nx, ny, Mach, gamma, pr_inf, rho_inf, theta, Uold)
    temp = U[1]
    for iter in 1:4
        U[iter] = U[iter] - 0.5 * globaldata.delta[idx][iter] * globaldata.flux_res[idx][iter]
    end
    if rk == 3
        for iter in 1:4
            U[iter] = U[iter] * 1/3 + Uold[iter] * 2/3
        end
    end
    U2_rot = U[2]
    U3_rot = U[3]
    U[2] = U2_rot*ny + U3_rot*nx
    U[3] = U3_rot*ny - U2_rot*nx
    globaldata.prim[idx][1] = U[1]
    temp = 1.0 / U[1]
    globaldata.prim[idx][2] = U[2]*temp
    globaldata.prim[idx][3] = U[3]*temp
    globaldata.prim[idx][4] = (0.4*U[4]) - (0.2*temp)*(U[2]*U[2] + U[3]*U[3])
    return nothing
end

function state_update_interior(globaldata, idx, max_res, ∑_res_sqr, U, Uold, rk)
    nx = globaldata.nx[idx]
    ny = globaldata.ny[idx]
    primitive_to_conserved(globaldata.prim[idx], nx, ny, U)
    primitive_to_conserved_old(globaldata.prim_old[idx], nx, ny, Uold)

    temp = U[1]
    for iter in 1:4
        U[iter] = U[iter] - 0.5 * globaldata.delta[idx][iter] * globaldata.flux_res[idx][iter]
    end
    if rk == 3
        for iter in 1:4
            U[iter] = U[iter] * 1/3 + Uold[iter] * 2/3
        end
    end
    U2_rot = U[2]
    U3_rot = U[3]
    U[2] = U2_rot*ny + U3_rot*nx
    U[3] = U3_rot*ny - U2_rot*nx
    res_sqr = (U[1] - temp)*(U[1] - temp)

    ∑_res_sqr[1] += res_sqr

    globaldata.prim[idx][1] = U[1]
    temp = 1.0 / U[1]
    globaldata.prim[idx][2] = U[2]*temp
    globaldata.prim[idx][3] = U[3]*temp
    globaldata.prim[idx][4] = (0.4*U[4]) - (0.2*temp)*(U[2]*U[2] + U[3]*U[3])
    return nothing
end

@inline function primitive_to_conserved(globaldata_prim, nx, ny, U)
    rho = globaldata_prim[1]
    U[1] = rho
    temp1::Float64 = rho * globaldata_prim[2]
    temp2::Float64 = rho * globaldata_prim[3]
    U[2] = temp1*ny - temp2*nx
    U[3] = temp1*nx + temp2*ny
    U[4] = 2.5*globaldata_prim[4] + 0.5*(temp1*temp1 + temp2*temp2)/rho
    return nothing
end

@inline function primitive_to_conserved_old(globaldata_prim_old, nx, ny, U)

    rho = globaldata_prim_old[1]
    U[1] = rho
    temp1 = rho * globaldata_prim_old[2]
    temp2 = rho * globaldata_prim_old[3]
    U[2] = temp1*ny - temp2*nx
    U[3] = temp1*nx + temp2*ny
    U[4] = 2.5*globaldata_prim_old[4] + 0.5*(temp1*temp1 + temp2*temp2)/rho
    return nothing
end

@inline function conserved_vector_Ubar(globaldata_prim, nx, ny, Mach, gamma, pr_inf, rho_inf, theta, Ubar)
    u1_inf = Mach*cos(theta)
    u2_inf = Mach*sin(theta)

    tx = ny
    ty = -nx

    u1_inf_rot = u1_inf*tx + u2_inf*ty
    u2_inf_rot = u1_inf*nx + u2_inf*ny

    temp1 = (u1_inf_rot * u1_inf_rot + u2_inf_rot*u2_inf_rot)
    e_inf = (pr_inf/(rho_inf*(gamma-1))) + 0.5 * (temp1)

    beta = (0.5 * rho_inf)/pr_inf
    S2 = u2_inf_rot * sqrt(beta)
    B2_inf = exp(-S2*S2)/(2*sqrt(Float64(pi)*beta))
    A2n_inf = 0.5 * (1 - SpecialFunctions.erf(S2))

    rho = globaldata_prim[1]
    u1 = globaldata_prim[2]
    u2 = globaldata_prim[3]
    pr = globaldata_prim[4]

    u1_rot = u1*tx + u2*ty
    u2_rot = u1*nx + u2*ny

    temp1 = (u1_rot*u1_rot + u2_rot*u2_rot)
    e = (pr/(rho*(gamma-1))) + 0.5*(temp1)

    beta = (rho)/(2*pr)
    S2 = u2_rot*sqrt(beta)
    B2 = exp(-S2*S2)/(2*sqrt(Float64(pi)*beta))
    A2p = 0.5*(1 + SpecialFunctions.erf(S2))

    Ubar[1] = (rho_inf*A2n_inf) + (rho*A2p)

    Ubar[2] = (rho_inf*u1_inf_rot*A2n_inf) + (rho*u1_rot*A2p)

    temp1 = rho_inf*(u2_inf_rot*A2n_inf - B2_inf)
    temp2 = rho*(u2_rot*A2p + B2)
    Ubar[3] = (temp1 + temp2)

    temp1 = (rho_inf*A2n_inf* e_inf - 0.5*rho_inf*u2_inf_rot*B2_inf)
    temp2 = (rho*A2p*e + 0.5*rho*u2_rot*B2)

    Ubar[4] = (temp1 + temp2)
    return nothing
end

@inline function conserved_vector_Ubar_old(globaldata_prim_old, nx, ny, Mach, gamma, pr_inf, rho_inf, theta, Ubar)
    u1_inf = Mach*cos(theta)
    u2_inf = Mach*sin(theta)

    tx = ny
    ty = -nx

    u1_inf_rot = u1_inf*tx + u2_inf*ty
    u2_inf_rot = u1_inf*nx + u2_inf*ny

    temp1 = (u1_inf_rot * u1_inf_rot + u2_inf_rot*u2_inf_rot)
    e_inf = (pr_inf/(rho_inf*(gamma-1))) + 0.5 * (temp1)

    beta = (0.5 * rho_inf)/pr_inf
    S2 = u2_inf_rot * sqrt(beta)
    B2_inf = exp(-S2*S2)/(2*sqrt(Float64(pi)*beta))
    A2n_inf = 0.5 * (1 - SpecialFunctions.erf(S2))

    rho = globaldata_prim_old[1]
    u1 = globaldata_prim_old[2]
    u2 = globaldata_prim_old[3]
    pr = globaldata_prim_old[4]

    u1_rot = u1*tx + u2*ty
    u2_rot = u1*nx + u2*ny

    temp1 = (u1_rot*u1_rot + u2_rot*u2_rot)
    e = (pr/(rho*(gamma-1))) + 0.5*(temp1)

    beta = (rho)/(2*pr)
    S2 = u2_rot*sqrt(beta)
    B2 = exp(-S2*S2)/(2*sqrt(Float64(pi)*beta))
    A2p = 0.5*(1 + SpecialFunctions.erf(S2))

    Ubar[1] = (rho_inf*A2n_inf) + (rho*A2p)

    Ubar[2] = (rho_inf*u1_inf_rot*A2n_inf) + (rho*u1_rot*A2p)

    temp1 = rho_inf*(u2_inf_rot*A2n_inf - B2_inf)
    temp2 = rho*(u2_rot*A2p + B2)
    Ubar[3] = (temp1 + temp2)

    temp1 = (rho_inf*A2n_inf* e_inf - 0.5*rho_inf*u2_inf_rot*B2_inf)
    temp2 = (rho*A2p*e + 0.5*rho*u2_rot*B2)

    Ubar[4] = (temp1 + temp2)
    return nothing
end