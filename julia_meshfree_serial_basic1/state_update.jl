import SpecialFunctions

function func_delta(globaldata, configData)
    cfl::Float64 = configData["core"]["cfl"]
    for (idx, store) in enumerate(globaldata)
        # TODO - Possible problem?
        min_delt = one(Float64)
        for itm in globaldata[idx].conn
            rho = globaldata[itm].prim[1]
            u1 = globaldata[itm].prim[2]
            u2 = globaldata[itm].prim[3]
            pr = globaldata[itm].prim[4]
            x_i = globaldata[idx].x
            y_i = globaldata[idx].y
            x_k = globaldata[itm].x
            y_k = globaldata[itm].y
            dist = hypot((x_k - x_i),(y_k - y_i))
            mod_u = hypot(u1,u2)
            delta_t = dist/(mod_u + 3*sqrt(pr/rho))
            delta_t *= cfl
            if min_delt > delta_t
                min_delt = delta_t
            end
        end
        globaldata[idx].delta = min_delt
    end
end

function state_update(globaldata, wallindices, outerindices, interiorindices, configData, iter, res_old)
    max_res = zero(Float64)
    sum_res_sqr = zero(Float64)
    U = zeros(Float64, 4)

    for itm in wallindices
        state_update_wall(globaldata, itm, max_res, sum_res_sqr, U)
    end

    for itm in outerindices
        state_update_outer(globaldata, itm, max_res, sum_res_sqr, U)
    end

    for itm in interiorindices
        state_update_interior(globaldata, itm, max_res, sum_res_sqr, U)
    end

    res_old = 0

    println("Iteration Number ", iter)
end

function state_update_wall(globaldata, itm, max_res, sum_res_sqr, U)
    nx = globaldata[itm].nx
    ny = globaldata[itm].ny
    primitive_to_conserved(globaldata, itm, nx, ny, U)
    temp = U[1]
    U -= (globaldata[itm].delta .* globaldata[itm].flux_res)
    U[3] = zero(Float64)
    U2_rot = U[2]
    U3_rot = U[3]
    U[2] = U2_rot*ny + U3_rot*nx
    U[3] = U3_rot*ny - U2_rot*nx
    res_sqr = (U[1] - temp)*(U[1] - temp)
    if res_sqr > max_res
        max_res = res_sqr
        max_res_point = itm
    end
    sum_res_sqr = sum_res_sqr + res_sqr
    globaldata[itm].prim[1] = U[1]
    temp = 1 / U[1]
    globaldata[itm].prim[2] = U[2]*temp
    globaldata[itm].prim[3] = U[3]*temp
    globaldata[itm].prim[4] = (0.4 *U[4]) - ((0.2 * temp)*(U[2] * U[2] + U[3] * U[3]))
end

function state_update_outer(globaldata, itm, max_res, sum_res_sqr, U)
    nx = globaldata[itm].nx
    ny = globaldata[itm].ny
    conserved_vector_Ubar(globaldata, itm, nx, ny, configData, U)
    temp = U[1]
    U = U - globaldata[itm].delta * globaldata[itm].flux_res
    U2_rot = U[2]
    U3_rot = U[3]
    U[2] = U2_rot*ny + U3_rot*nx
    U[3] = U3_rot*ny - U2_rot*nx
    globaldata[itm].prim[1] = U[1]
    temp = 1 / U[1]
    globaldata[itm].prim[2] = U[2]*temp
    globaldata[itm].prim[3] = U[3]*temp
    globaldata[itm].prim[4] = (0.4*U[4]) - (0.2*temp)*(U[2]*U[2] + U[3]*U[3])
end

function state_update_interior(globaldata, itm, max_res, sum_res_sqr, U)
    nx = globaldata[itm].nx
    ny = globaldata[itm].ny
    primitive_to_conserved(globaldata, itm, nx, ny, U)
    temp = U[1]
    U = U - globaldata[itm].delta .* globaldata[itm].flux_res

    U2_rot = U[2]
    U3_rot = U[3]
    U[2] = U2_rot*ny + U3_rot*nx
    U[3] = U3_rot*ny - U2_rot*nx

    globaldata[itm].prim[1] = U[1]
    temp = 1 / U[1]
    globaldata[itm].prim[2] = U[2]*temp
    globaldata[itm].prim[3] = U[3]*temp
    globaldata[itm].prim[4] = (0.4*U[4]) - (0.2*temp)*(U[2]*U[2] + U[3]*U[3])
end

@inline function primitive_to_conserved(globaldata, itm, nx, ny, U)
    rho = globaldata[itm].prim[1]
    U[1] = rho
    temp1 = rho*globaldata[itm].prim[2]
    temp2 = rho*globaldata[itm].prim[3]
    U[2] = temp1*ny - temp2*nx
    U[3] = temp1*nx + temp2*ny
    U[4] = 2.5*globaldata[itm].prim[4] + 0.5*(temp1*temp1 + temp2*temp2)/rho
end

@inline function conserved_vector_Ubar(globaldata, itm, nx, ny, configData, Ubar)
    Mach::Float64 = configData["core"]["mach"]
    gamma::Float64 = configData["core"]["gamma"]
    pr_inf::Float64 = configData["core"]["pr_inf"]
    rho_inf::Float64 = configData["core"]["rho_inf"]
    theta = calculateTheta(configData)

    u1_inf = Mach*cos(theta)
    u2_inf = Mach*sin(theta)

    tx = ny
    ty = -nx

    u1_inf_rot = u1_inf*tx + u2_inf*ty
    u2_inf_rot = u1_inf*nx + u2_inf*ny

    temp1 = (u1_inf_rot*u1_inf_rot + u2_inf_rot*u2_inf_rot)
    e_inf = pr_inf/(rho_inf*(gamma-1)) + 0.5*(temp1)

    beta = (0.5*rho_inf)/pr_inf
    S2 = u2_inf_rot*sqrt(beta)
    B2_inf = exp(-S2*S2)/(2*sqrt(pi*beta))
    A2n_inf = 0.5*(1 - SpecialFunctions.erf(S2))

    rho = globaldata[itm].prim[1]
    u1 = globaldata[itm].prim[2]
    u2 = globaldata[itm].prim[3]
    pr = globaldata[itm].prim[4]

    u1_rot = u1*tx + u2*ty
    u2_rot = u1*nx + u2*ny

    temp1 = (u1_rot*u1_rot + u2_rot*u2_rot)
    e = pr/(rho*(gamma-1)) + 0.5*(temp1)

    beta = (rho)/(2*pr)
    S2 = u2_rot*sqrt(beta)
    B2 = exp(-S2*S2)/(2*sqrt(pi*beta))
    A2p = 0.5*(1+ SpecialFunctions.erf(S2))

    Ubar[1] = (rho_inf*A2n_inf) + (rho*A2p)

    Ubar[2] = (rho_inf*u1_inf_rot*A2n_inf) + (rho*u1_rot*A2p)

    temp1 = rho_inf*(u2_inf_rot*A2n_inf - B2_inf)
    temp2 = rho*(u2_rot*A2p + B2)
    Ubar[3] = temp1 + temp2

    temp1 = (rho_inf*A2n_inf*e_inf - 0.5*rho_inf*u2_inf_rot*B2_inf)
    temp2 = (rho*A2p*e + 0.5*rho*u2_rot*B2)

    Ubar[4] = temp1 + temp2
end
