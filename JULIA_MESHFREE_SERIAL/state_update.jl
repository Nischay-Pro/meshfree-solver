import SpecialFunctions
function func_delta(globaldata, configData)
    cfl = configData["core"]["cfl"]::Float64
    for (idx, store) in enumerate(globaldata)
        # TODO - Possible problem?
        min_delt = one(Float64)
        for itm in globaldata[idx].conn
            x_i = globaldata[idx].x
            y_i = globaldata[idx].y
            x_k = globaldata[itm].x
            y_k = globaldata[itm].y
            dist = hypot((x_k - x_i),(y_k - y_i))
            mod_u = hypot(globaldata[itm].prim[2],globaldata[itm].prim[3])
            delta_t = dist/(mod_u + 3*sqrt(globaldata[itm].prim[4]/globaldata[itm].prim[1]))
            delta_t *= cfl
            if min_delt > delta_t
                min_delt = delta_t
            end
        end
        globaldata[idx].delta = min_delt
        globaldata[idx].prim_old = globaldata[idx].prim
    end
    return nothing
end

function state_update(globaldata, wallindices, outerindices, interiorindices, configData, iter, res_old, rk, numPoints)
    max_res = zero(Float64)
    sum_res_sqr = zeros(Float64, 1)
    U = zeros(Float64, 4)
    Uold = zeros(Float64, 4)
    # println("Prim1.01a")
    # println(IOContext(stdout, :compact => false), globaldata[1].prim)
    for itm in wallindices
        fill!(U, 0.0)
        state_update_wall(globaldata, itm, max_res, sum_res_sqr, U, Uold, rk)
        # if itm == 3
        #     println(sum_res_sqr)
        # end
    end

    # println("Prim1.01b")
    # println(IOContext(stdout, :compact => false), globaldata[1].prim)

    for itm in outerindices
        fill!(U, 0.0)
        state_update_outer(globaldata, configData, itm, max_res, sum_res_sqr, U, Uold, rk)
    end

    for itm in interiorindices
        fill!(U, 0.0)
        # if itm == 1
        #     println("Prim1.01c")
        #     println(IOContext(stdout, :compact => false), globaldata[itm].prim)
        # end
        state_update_interior(globaldata, itm, max_res, sum_res_sqr, U, Uold, rk)
    end
    # println(sum_res_sqr[1])
    # println("The length is ", length(globaldata))
    res_new = sqrt(sum_res_sqr[1])/ length(globaldata)
    residue = 0
    # println(res_old)
    if iter <= 2
        res_old[1] = res_new
        residue = 0
    else
        residue = log10(res_new/res_old[1])
    end
    # if rk == 4
    #     print(" ", residue, " ")
    # end
    # open("residue_" * string(numPoints) * ".txt", "a+") do residue_io
    #     @printf(residue_io, "%d %s\n", iter, residue)
    # end

    # res_old = 0

    return  nothing
end

function state_update_wall(globaldata, itm, max_res, sum_res_sqr, U, Uold, rk)
    nx = globaldata[itm].nx
    ny = globaldata[itm].ny
    # if itm == 2
    #     println("Prim1.01a2.1")
    #     println(IOContext(stdout, :compact => false), globaldata[1].prim)
    # end
    primitive_to_conserved(globaldata, itm, nx, ny, U)
    # if itm == 2
    #     println("Prim1.01a2.2")
    #     println(IOContext(stdout, :compact => false), globaldata[1].prim)
    # end
    temp = U[1]
    U = @. U - (globaldata[itm].delta .* globaldata[itm].flux_res)
    if rk == 3
        primitive_to_conserved_old(globaldata, itm, nx, ny, Uold)
        U = @. U * 1/3 + Uold * 2/3
    end
    U[3] = zero(Float64)
    U2_rot = U[2]
    U3_rot = U[3]
    U[2] = U2_rot*ny + U3_rot*nx
    U[3] = U3_rot*ny - U2_rot*nx
    res_sqr = (U[1] - temp)*(U[1] - temp)
    # if itm == 2
    #     println("Prim1.01a2.3")
    #     println(IOContext(stdout, :compact => false), globaldata[1].prim)
    # end
    # if res_sqr > max_res
    #     max_res = res_sqr
    #     max_res_point = itm
    # end
    sum_res_sqr[1] += res_sqr
    globaldata[itm].prim[1] = U[1]
    temp = 1.0 / U[1]
    globaldata[itm].prim[2] = U[2]*temp
    globaldata[itm].prim[3] = U[3]*temp
    globaldata[itm].prim[4] = (0.4*U[4]) - ((0.2 * temp) * (U[2] * U[2] + U[3] * U[3]))
    # if itm == 2
    #     println("Prim1.01a2.5")
    #     println(IOContext(stdout, :compact => false), globaldata[1].prim)
    # end
end

@inline function state_update_outer(globaldata, configData, itm, max_res, sum_res_sqr, U, Uold, rk)
    nx = globaldata[itm].nx
    ny = globaldata[itm].ny
    conserved_vector_Ubar(globaldata, itm, nx, ny, configData, U)
    temp = U[1]
    U = @. U - globaldata[itm].delta * globaldata[itm].flux_res
    if rk == 3
        conserved_vector_Ubar_old(globaldata, itm, nx, ny, configData, Uold)
        U = @. U * 1/3 + Uold * 2/3
    end
    U2_rot = U[2]
    U3_rot = U[3]
    U[2] = U2_rot*ny + U3_rot*nx
    U[3] = U3_rot*ny - U2_rot*nx
    globaldata[itm].prim[1] = U[1]
    temp = 1.0 / U[1]
    globaldata[itm].prim[2] = U[2]*temp
    globaldata[itm].prim[3] = U[3]*temp
    globaldata[itm].prim[4] = (0.4*U[4]) - (0.2*temp)*(U[2]*U[2] + U[3]*U[3])
end

@inline function state_update_interior(globaldata, itm, max_res, sum_res_sqr, U, Uold, rk)
    nx = globaldata[itm].nx
    ny = globaldata[itm].ny
    primitive_to_conserved(globaldata, itm, nx, ny, U)
    # if itm == 1
    #     println("Prim1.11")
    #     println(IOContext(stdout, :compact => false), globaldata[itm].prim)
    # end
    # if itm == 1
    #     println(IOContext(stdout, :compact => false), U)
    #     # println(IOContext(stdout, :compact => false), temp)
    # end
    temp = U[1]
    U = @. U - globaldata[itm].delta .* globaldata[itm].flux_res
    if rk == 3
        primitive_to_conserved_old(globaldata, itm, nx, ny, Uold)
        U = @. U * 1/3 + Uold * 2/3
    end
    U2_rot = U[2]
    U3_rot = U[3]
    U[2] = U2_rot*ny + U3_rot*nx
    U[3] = U3_rot*ny - U2_rot*nx
    res_sqr = (U[1] - temp)*(U[1] - temp)
    # if itm == 2
    #     println("Prim1.01a2.3")
    #     println(IOContext(stdout, :compact => false), globaldata[1].prim)
    # end
    # if res_sqr > max_res
    #     max_res = res_sqr
    #     max_res_point = itm
    # end
    sum_res_sqr[1] += res_sqr
    # if itm == 1
    #     println(IOContext(stdout, :compact => false), U)
    #     println(IOContext(stdout, :compact => false), temp)
    # end
    globaldata[itm].prim[1] = U[1]
    temp = 1.0 / U[1]
    globaldata[itm].prim[2] = U[2]*temp
    globaldata[itm].prim[3] = U[3]*temp
    globaldata[itm].prim[4] = (0.4*U[4]) - (0.2*temp)*(U[2]*U[2] + U[3]*U[3])
end

@inline function primitive_to_conserved(globaldata, itm, nx, ny, U)
    # if itm == 1
    #     println("Prim1.1")
    #     println(IOContext(stdout, :compact => false), globaldata[itm].prim)
    # end
    rho = globaldata[itm].prim[1]
    U[1] = rho
    temp1::Float64 = rho * globaldata[itm].prim[2]
    temp2::Float64 = rho * globaldata[itm].prim[3]
    U[2] = temp1*ny - temp2*nx
    U[3] = temp1*nx + temp2*ny
    U[4] = 2.5*globaldata[itm].prim[4] + 0.5*(temp1^2 + temp2^2)/rho
    # if itm == 1
    #     println("U")
    #     println(IOContext(stdout, :compact => false), U)
    # end
end

@inline function primitive_to_conserved_old(globaldata, itm, nx, ny, U)
    # if itm == 1
    #     println("Prim1.1")
    #     println(IOContext(stdout, :compact => false), globaldata[itm].prim)
    # end
    rho = globaldata[itm].prim_old[1]
    U[1] = rho
    temp1::Float64 = rho * globaldata[itm].prim_old[2]
    temp2::Float64 = rho * globaldata[itm].prim_old[3]
    U[2] = temp1*ny - temp2*nx
    U[3] = temp1*nx + temp2*ny
    U[4] = 2.5*globaldata[itm].prim_old[4] + 0.5*(temp1^2 + temp2^2)/rho
    # if itm == 1
    #     println("U")
    #     println(IOContext(stdout, :compact => false), U)
    # end
end

@inline function conserved_vector_Ubar(globaldata, itm, nx, ny, configData, Ubar)
    Mach::Float64 = configData["core"]["mach"]::Float64
    gamma::Float64 = configData["core"]["gamma"]::Float64
    pr_inf::Float64 = configData["core"]["pr_inf"]::Float64
    rho_inf::Float64 = configData["core"]["rho_inf"]::Float64
    theta = calculateTheta(configData)

    u1_inf::Float64 = Mach*cos(theta)
    u2_inf::Float64 = Mach*sin(theta)

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

    rho = globaldata[itm].prim[1]
    u1 = globaldata[itm].prim[2]
    u2 = globaldata[itm].prim[3]
    pr = globaldata[itm].prim[4]

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
end

@inline function conserved_vector_Ubar_old(globaldata, itm, nx, ny, configData, Ubar)
    Mach::Float64 = configData["core"]["mach"]::Float64
    gamma::Float64 = configData["core"]["gamma"]::Float64
    pr_inf::Float64 = configData["core"]["pr_inf"]::Float64
    rho_inf::Float64 = configData["core"]["rho_inf"]::Float64
    theta = calculateTheta(configData)

    u1_inf::Float64 = Mach*cos(theta)
    u2_inf::Float64 = Mach*sin(theta)

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

    rho = globaldata[itm].prim_old[1]
    u1 = globaldata[itm].prim_old[2]
    u2 = globaldata[itm].prim_old[3]
    pr = globaldata[itm].prim_old[4]

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
end