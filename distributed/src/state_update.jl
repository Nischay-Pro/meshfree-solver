import SpecialFunctions
function func_delta(loc_globaldata, loc_ghost_holder, cfl)
    dist_length = length(loc_globaldata)
    for idx in 1:dist_length
        min_delt = one(Float64)
        for itm in loc_globaldata[idx].conn
            if itm == zero(Float64)
                break
            end
            if itm <= dist_length
                globaldata_itm = loc_globaldata[itm]
            else
                globaldata_itm = loc_ghost_holder[1][itm]
            end
            x_i = loc_globaldata[idx].x
            y_i = loc_globaldata[idx].y
            x_k = globaldata_itm.x
            y_k = globaldata_itm.y
            dist = hypot((x_k - x_i),(y_k - y_i))
            mod_u = hypot(globaldata_itm.prim[2],globaldata_itm.prim[3])
            delta_t = dist/(mod_u + 3*sqrt(globaldata_itm.prim[4]/globaldata_itm.prim[1]))
            delta_t *= cfl
            if min_delt > delta_t
                min_delt = delta_t
            end
        end
        loc_globaldata[idx].delta = min_delt
        @. loc_globaldata[idx].prim_old = loc_globaldata[idx].prim
    end
    return nothing
end

function state_update(loc_globaldata, loc_prim, iter, res_old, res_new, rk, U, Uold, main_store)
    ∑_res_sqr = zeros(Float64, 1)
    Mach = main_store[58] 
    gamma = main_store[59] 
    pr_inf = main_store[60] 
    rho_inf = main_store[61] 
    theta = main_store[62]

    for (idx, _) in enumerate(loc_globaldata)
        if loc_globaldata[idx].flag_1 == 0
            fill!(U, zero(Float64))
            state_update_wall(loc_globaldata, loc_prim, idx, ∑_res_sqr, U, Uold, rk)
        elseif loc_globaldata[idx].flag_1 == 2
            fill!(U, zero(Float64))
            state_update_outer(loc_globaldata, loc_prim, idx, Mach, gamma, pr_inf, rho_inf, theta, ∑_res_sqr, U, Uold, rk)
        elseif loc_globaldata[idx].flag_1 == 1
            fill!(U, zero(Float64))
            state_update_interior(loc_globaldata, loc_prim, idx, ∑_res_sqr, U, Uold, rk)
        end
    end

    res_new[1] = ∑_res_sqr[1]
    if iter <= 2
        res_old[1] = res_new[1]
    end
    return nothing
end

function state_update_wall(globaldata, loc_prim, idx, ∑_res_sqr, U, Uold, rk)
    nx = globaldata[idx].nx
    ny = globaldata[idx].ny
    # if idx == 2
    #     println("Prim1.01a2.1")
    #     println(IOContext(stdout, :compact => false), globaldata[1].prim)
    # end
    primitive_to_conserved(globaldata[idx].prim, idx, nx, ny, U)
    primitive_to_conserved(globaldata[idx].prim_old, idx, nx, ny, Uold)
    temp = U[1]
    @. U = U - 0.5 * (globaldata[idx].delta .* globaldata[idx].flux_res)
    if rk == 3
        @. U = U * 1/3 + Uold * 2/3
    end
    U[3] = zero(Float64)
    U2_rot = U[2]
    U3_rot = U[3]
    U[2] = U2_rot*ny + U3_rot*nx
    U[3] = U3_rot*ny - U2_rot*nx
    res_sqr = (U[1] - temp)*(U[1] - temp)
    ∑_res_sqr[1] += res_sqr
    globaldata[idx].prim[1] = U[1]
    temp = 1.0 / U[1]
    globaldata[idx].prim[2] = U[2]*temp
    globaldata[idx].prim[3] = U[3]*temp
    globaldata[idx].prim[4] = (0.4*U[4]) - ((0.2 * temp) * (U[2] * U[2] + U[3] * U[3]))
    for iter in 1:4
        loc_prim[idx].prim = setindex(loc_prim[idx].prim, globaldata[idx].prim[iter], iter)
    end
    return nothing
end

@inline function state_update_outer(globaldata, loc_prim, idx, Mach, gamma, pr_inf, rho_inf, theta, ∑_res_sqr, U, Uold, rk)
    nx = globaldata[idx].nx
    ny = globaldata[idx].ny
    conserved_vector_Ubar(globaldata[idx].prim, idx, nx, ny, Mach, gamma, pr_inf, rho_inf, theta, U)
    conserved_vector_Ubar(globaldata[idx].prim_old, idx, nx, ny, Mach, gamma, pr_inf, rho_inf, theta, Uold)
    temp = U[1]
    @. U = U - 0.5 * globaldata[idx].delta * globaldata[idx].flux_res
    if rk == 3
        @. U = U * 1/3 + Uold * 2/3
    end
    U2_rot = U[2]
    U3_rot = U[3]
    U[2] = U2_rot*ny + U3_rot*nx
    U[3] = U3_rot*ny - U2_rot*nx
    globaldata[idx].prim[1] = U[1]
    temp = 1.0 / U[1]
    globaldata[idx].prim[2] = U[2]*temp
    globaldata[idx].prim[3] = U[3]*temp
    globaldata[idx].prim[4] = (0.4*U[4]) - (0.2*temp)*(U[2]*U[2] + U[3]*U[3])
    for iter in 1:4
        loc_prim[idx].prim = setindex(loc_prim[idx].prim, globaldata[idx].prim[iter], iter)
    end
    return nothing
end

@inline function state_update_interior(globaldata, loc_prim,idx, ∑_res_sqr, U, Uold, rk)
    nx = globaldata[idx].nx
    ny = globaldata[idx].ny
    primitive_to_conserved(globaldata[idx].prim, idx, nx, ny, U)
    primitive_to_conserved(globaldata[idx].prim_old, idx, nx, ny, Uold)

    temp = U[1]
    @. U = U - 0.5 * globaldata[idx].delta .* globaldata[idx].flux_res
    if rk == 3
        @. U =U * 1/3 + Uold * 2/3
    end
    U2_rot = U[2]
    U3_rot = U[3]
    U[2] = U2_rot*ny + U3_rot*nx
    U[3] = U3_rot*ny - U2_rot*nx
    res_sqr = (U[1] - temp)*(U[1] - temp)
    ∑_res_sqr[1] += res_sqr
    globaldata[idx].prim[1] = U[1]
    temp = 1.0 / U[1]
    globaldata[idx].prim[2] = U[2]*temp
    globaldata[idx].prim[3] = U[3]*temp
    globaldata[idx].prim[4] = (0.4*U[4]) - (0.2*temp)*(U[2]*U[2] + U[3]*U[3])
    for iter in 1:4
        loc_prim[idx].prim = setindex(loc_prim[idx].prim, globaldata[idx].prim[iter], iter)
    end
    return nothing
end

@inline function primitive_to_conserved(globaldata_prim, idx, nx, ny, U)

    rho = globaldata_prim[1]
    U[1] = rho
    temp1::Float64 = rho * globaldata_prim[2]
    temp2::Float64 = rho * globaldata_prim[3]
    U[2] = temp1*ny - temp2*nx
    U[3] = temp1*nx + temp2*ny
    U[4] = 2.5*globaldata_prim[4] + 0.5*(temp1^2 + temp2^2)/rho
    return nothing
end

@inline function conserved_vector_Ubar(globaldata_prim, idx, nx, ny, Mach, gamma, pr_inf, rho_inf, theta, Ubar)
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
