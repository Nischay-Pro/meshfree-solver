import SpecialFunctions

function func_delta(globaldata, configData)
    cfl = configData["core"]["cfl"]
    for (idx, store) in enumerate(globaldata)
        if idx > 0
            # TODO - Possible problem?
            min_delt = 1
            for itm in globaldata[idx].conn
                rho = globaldata[itm].prim[1]
                u1 = globaldata[itm].prim[2]
                u2 = globaldata[itm].prim[3]
                pr = globaldata[itm].prim[4]

                x_i = globaldata[idx].x
                y_i = globaldata[idx].y

                x_k = globaldata[itm].x
                y_k = globaldata[itm].y

                dist = (x_k - x_i)*(x_k - x_i) + (y_k - y_i)*(y_k - y_i)
                dist = sqrt(dist)

                mod_u = hypot(u1,u2)

                delta_t = dist/(mod_u + 3*sqrt(pr/rho))

                delta_t = cfl*delta_t

                if min_delt > delta_t
                    min_delt = delta_t
                end
            end
            globaldata[idx].delta = min_delt
        end
    end
    return globaldata
end

function state_update(globaldata, wallindices, outerindices, interiorindices, configData, iter, res_old)
    max_res = 0
    sum_res_sqr = 0
    U = zeros(Float64, 4)
    # print(typeof(globaldata[1].prim[1]))
    for itm in wallindices
        nx = globaldata[itm].nx
        ny = globaldata[itm].ny
        U = primitive_to_conserved(globaldata, itm, nx, ny)
        # if itm == 77
        #     println("=====================")
        #     println(nx)
        #     println(ny)
        #     println(U)
        # end
        temp = U[1]
        U = U - (globaldata[itm].delta .* globaldata[itm].flux_res)
        # if itm == 77
        #     println(globaldata[itm].delta, " ", globaldata[itm].flux_res)
        # end
        U[3] = 0

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

        # if itm == 100
        #     print(tempU[1])
        #     print(tempU[2])
        #     print(tempU[3])
        #     print(tempU[4])
        # end
    end
    # print(wallindices)
    # print(globaldata[1].prim[1])
    # print("\n 444a")

    for itm in outerindices
        nx = globaldata[itm].nx
        ny = globaldata[itm].ny
        U = conserved_vector_Ubar(globaldata, itm, nx, ny, configData)
        temp = U[1]

        # TODO - Check this calculation
        # U = np.array(U) - globaldata[itm].delta * np.array(globaldata[itm].flux_res)
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

    # print(outerindices)
    # print(globaldata[1].prim[1])
    # print("\n 444b")

    for itm in interiorindices
        nx = globaldata[itm].nx
        ny = globaldata[itm].ny
        U = primitive_to_conserved(globaldata, itm, nx, ny)
        temp = U[1]

        # if itm == 1
        #     print(U)
        #     print(globaldata[itm].delta)
        #     print(globaldata[itm].flux_res)
        # end

        U = U - globaldata[itm].delta .* globaldata[itm].flux_res

        U2_rot = U[2]
        U3_rot = U[3]
        U[2] = U2_rot*ny + U3_rot*nx
        U[3] = U3_rot*ny - U2_rot*nx

        # res_sqr = (U[0] - temp)*(U[0] - temp)

        # if res_sqr > max_res
        #     max_res = res_sqr
        #     max_res_point = itm

        # sum_res_sqr = sum_res_sqr + res_sqr

        globaldata[itm].prim[1] = U[1]
        temp = 1 / U[1]
        globaldata[itm].prim[2] = U[2]*temp
        globaldata[itm].prim[3] = U[3]*temp
        globaldata[itm].prim[4] = (0.4*U[4]) - (0.2*temp)*(U[2]*U[2] + U[3]*U[3])
    end

    # print(globaldata[1].prim[1])
    # print("\n 444c")

    res_old = 0

    # res_new = sqrt(sum_res_sqr)/ len(globaldata)

    # if iter <= 2
    #     res_old = res_new
    #     residue = 0
    # else
    #     residue = log10(res_new/res_old)

    # with open('residue', 'a') as the_file
    #     the_file.write("%i %f" % (iter, residue))

    println("Iteration Number ", iter)
    # # print("Residue ", residue)

    return globaldata, res_old
end

function primitive_to_conserved(globaldata, itm, nx, ny)
    U = []

    rho = globaldata[itm].prim[1]
    push!(U, rho)
    temp1 = rho*globaldata[itm].prim[2]
    temp2 = rho*globaldata[itm].prim[3]

    push!(U, temp1*ny - temp2*nx)
    push!(U, temp1*nx + temp2*ny)
    push!(U, 2.5*globaldata[itm].prim[4] + 0.5*(temp1*temp1 + temp2*temp2)/rho)

    # if itm == 1
    #     print("\n")
    #     print(U)
    #     print("     ABCD")
    # end
    return U
end

function conserved_vector_Ubar(globaldata, itm, nx, ny, configData)
    Mach = configData["core"]["mach"]
    gamma = configData["core"]["gamma"]
    pr_inf = configData["core"]["pr_inf"]
    rho_inf = configData["core"]["rho_inf"]

    Ubar = []

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

    push!(Ubar, (rho_inf*A2n_inf) + (rho*A2p))

    push!(Ubar, (rho_inf*u1_inf_rot*A2n_inf) + (rho*u1_rot*A2p))

    temp1 = rho_inf*(u2_inf_rot*A2n_inf - B2_inf)
    temp2 = rho*(u2_rot*A2p + B2)
    push!(Ubar, temp1 + temp2)

    temp1 = (rho_inf*A2n_inf*e_inf - 0.5*rho_inf*u2_inf_rot*B2_inf)
    temp2 = (rho*A2p*e + 0.5*rho*u2_rot*B2)

    push!(Ubar, temp1 + temp2)

    return Ubar
end
