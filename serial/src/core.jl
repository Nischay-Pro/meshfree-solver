function getInitialPrimitive(configData)
    rho_inf::Float64 = configData["core"]["rho_inf"]
    mach::Float64 = configData["core"]["mach"]
    machcos::Float64 = mach * cos(calculateTheta(configData))
    machsin::Float64 = mach * sin(calculateTheta(configData))
    pr_inf::Float64 = configData["core"]["pr_inf"]
    primal = [rho_inf, machcos, machsin, pr_inf]
    return primal
end

function getInitialPrimitive2(configData)
    dataman = open("prim_soln_clean")
    data = read(dataman, String)
    data1 = split(data, "\n")
    finaldata = Array{Array{Float64,1},1}(undef, 0)
    for (idx,itm) in enumerate(data1)
        # try
        da = split(itm)
        da1 = parse.(Float64, da)
        push!(finaldata, da1)
    end
    close(dataman)
    return finaldata
end

function placeNormals(globaldata, idx, configData, interior, wall, outer)
    flag = globaldata[idx].flag_1
    if flag == wall || flag == outer
        currpt = getxy(globaldata[idx])
        leftpt = globaldata[idx].left
        leftpt = getxy(globaldata[leftpt])
        rightpt = globaldata[idx].right
        rightpt = getxy(globaldata[rightpt])
        normals = calculateNormals(leftpt, rightpt, currpt[1], currpt[2])
        setNormals(globaldata[idx], normals)
    elseif flag == interior
        setNormals(globaldata[idx], (0,1))
    else
        @warn "Illegal Point Type"
    end
end

function calculateNormals(left, right, mx, my)
    lx = left[1]
    ly = left[2]

    rx = right[1]
    ry = right[2]

    nx1 = my - ly
    nx2 = ry - my

    ny1 = mx - lx
    ny2 = rx - mx

    nx = 0.5*(nx1 + nx2)
    ny = 0.5*(ny1 + ny2)

    det = hypot(nx, ny)

    nx = -nx/det
    ny = ny/det

    return (nx,ny)
end

function calculateConnectivity(globaldata, idx)
    ptInterest = globaldata[idx]
    currx = ptInterest.x
    curry = ptInterest.y
    nx = ptInterest.nx
    ny = ptInterest.ny

    flag = ptInterest.flag_1

    xpos_conn,xneg_conn,ypos_conn,yneg_conn = Array{Int32,1}(undef, 0),Array{Int32,1}(undef, 0),Array{Int32,1}(undef, 0),Array{Int32,1}(undef, 0)

    tx = ny
    ty = -nx

    for itm in ptInterest.conn
        itmx = globaldata[itm].x
        itmy = globaldata[itm].y

        delx = itmx - currx
        dely = itmy - curry

        dels = delx*tx + dely*ty
        deln = delx*nx + dely*ny
        if dels <= 0.0
            push!(xpos_conn, itm)
        end
        if dels >= 0.0
            push!(xneg_conn, itm)
        end
        if flag == 1
            if deln <= 0.0
                push!(ypos_conn, itm)
            end
            if deln >= 0.0
                push!(yneg_conn, itm)
            end
        elseif flag == 0
            push!(yneg_conn, itm)
        elseif flag == 2
            push!(ypos_conn, itm)
        end
    end
    return (xpos_conn, xneg_conn, ypos_conn, yneg_conn)
end

function getPointDetails(globaldata, point_index)
    println("")
    println(IOContext(stdout, :compact => false), "Q is", globaldata[point_index].q)
    println(IOContext(stdout, :compact => false), "DQ is", globaldata[point_index].dq)
    println(IOContext(stdout, :compact => false), "Prim is", globaldata[point_index].prim)
    println(IOContext(stdout, :compact => false), "Flux Res is", globaldata[point_index].flux_res)
    # println(IOContext(stdout, :compact => false), "MaxQ is", globaldata[point_index].max_q)
    # println(IOContext(stdout, :compact => false), "MinQ is", globaldata[point_index].min_q)
    println(IOContext(stdout, :compact => false), "Prim Old is", globaldata[point_index].prim_old)
    # println(IOContext(stdout, :compact => false), "Delta is", globaldata[point_index].delta)
end

function fpi_solver(iter, globaldata, configData, res_old, numPoints, main_store, tempdq)
    # println(IOContext(stdout, :compact => false), globaldata[3].prim)
    # print(" 111\n")
    if iter == 1
        println("Starting FuncDelta")
    end
    func_delta(globaldata, configData)

    phi_i = @view main_store[1:4]
	phi_k = @view main_store[5:8]
	G_i = @view main_store[9:12]
    G_k = @view main_store[13:16]
	result = @view main_store[17:20]
	qtilde_i = @view main_store[21:24]
	qtilde_k = @view main_store[25:28]
	Gxp = @view main_store[29:32]
	Gxn = @view main_store[33:36]
	Gyp = @view main_store[37:40]
	Gyn = @view main_store[41:44]
    sum_delx_delf = @view main_store[45:48]
    sum_dely_delf = @view main_store[49:52]

    power::Float64 = configData["core"]["power"]

    print("Iteration Number ", iter, " ")
    # getPointDetails(globaldata, 3)
    for rk in 1:4
        q_variables(globaldata, configData)
        # println("=========")
        # if iter == 1
            # println("Starting QVar")
        # end
        # @timeit to "nest 2" begin
        @timeit to "nest 2" begin
            q_var_derivatives(globaldata, power)
        end
        for inner_iters in 1:3
            @profile q_var_derivatives_innerloop(globaldata, power, tempdq)
        end
        # end
        # end
        # getPointDetails(globaldata, 3)
        # println(IOContext(stdout, :compact => false), globaldata[3].prim)
        # if iter == 1
            # println("Starting Calflux")
        # end

        @timeit to "nest 3" begin
        cal_flux_residual(globaldata, configData, Gxp, Gxn, Gyp, Gyn, phi_i, phi_k, G_i, G_k,
                result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf)
        end
        # getPointDetails(globaldata, 3)
        # println(IOContext(stdout, :compact => false), globaldata[3].prim)
        # residue = 0
        # if iter == 1
            # println("Starting StateUpdate")
        # end
        state_update(globaldata, configData, iter, res_old, rk, numPoints)
        # getPointDetails(globaldata, 3)
    end

    # println(IOContext(stdout, :compact => false), globaldata[3].prim)
    # residue = res_old
    return nothing
end

function q_variables(globaldata::Array{Point,1}, configData)
    for (idx, itm) in enumerate(globaldata)
        rho = itm.prim[1]
        u1 = itm.prim[2]
        u2 = itm.prim[3]
        pr = itm.prim[4]

        beta = 0.5 * (rho / pr)
        globaldata[idx].q[1] = log(rho) + log(beta) * 2.5 - (beta * ((u1 * u1) + (u2 * u2)))
        two_times_beta = 2.0 * beta
        globaldata[idx].q[2] = (two_times_beta * u1)
        globaldata[idx].q[3] = (two_times_beta * u2)
        globaldata[idx].q[4] = -two_times_beta

    end
end

function q_var_derivatives(globaldata::Array{Point,1}, power)
    sum_delx_delq = zeros(Float64, 4)
    sum_dely_delq = zeros(Float64, 4)
    for (idx, itm) in enumerate(globaldata)
        x_i = itm.x
        y_i = itm.y
        sum_delx_sqr = zero(Float64)
        sum_dely_sqr = zero(Float64)
        sum_delx_dely = zero(Float64)
        fill!(sum_delx_delq, zero(Float64))
        fill!(sum_dely_delq, zero(Float64))

        @. itm.max_q = itm.q
        @. itm.min_q = itm.q

        for conn in itm.conn
            x_k = globaldata[conn].x
            y_k = globaldata[conn].y
            delx = x_k - x_i
            dely = y_k - y_i
            dist = hypot(delx, dely)
            weights = dist ^ power
            sum_delx_sqr += ((delx * delx) * weights)
            sum_dely_sqr += ((dely * dely) * weights)
            sum_delx_dely += ((delx * dely) * weights)

            @. sum_delx_delq += (weights * delx * (globaldata[conn].q - globaldata[idx].q))
            @. sum_dely_delq += (weights * dely * (globaldata[conn].q - globaldata[idx].q))
            for i in 1:4
                if globaldata[idx].max_q[i] < globaldata[conn].q[i]
                    globaldata[idx].max_q[i] = globaldata[conn].q[i]
                end
                if globaldata[idx].min_q[i] > globaldata[conn].q[i]
                    globaldata[idx].min_q[i] = globaldata[conn].q[i]
                end
            end
        end
        det = (sum_delx_sqr * sum_dely_sqr) - (sum_delx_dely * sum_delx_dely)
        one_by_det = 1.0 / det
        for iter in 1:4
            itm.dq[1][iter] = one_by_det * (sum_delx_delq[iter] * sum_dely_sqr - sum_dely_delq[iter] * sum_delx_dely)
            itm.dq[2][iter] = one_by_det * (sum_dely_delq[iter] * sum_delx_sqr - sum_delx_delq[iter] * sum_delx_dely)
        end
    end
    # println(IOContext(stdout, :compact => false), globaldata[3].dq)
    # println(IOContext(stdout, :compact => false), globaldata[3].max_q)
    # println(IOContext(stdout, :compact => false), globaldata[3].min_q)
    return nothing
end

function q_var_derivatives_innerloop(globaldata::Array{Point,1}, power, tempdq)
    sum_delx_delq = zeros(Float64, 4)
    sum_dely_delq = zeros(Float64, 4)
    qi_tilde = zeros(Float64, 4)
    qk_tilde = zeros(Float64, 4)
    for (idx, itm) in enumerate(globaldata)
        x_i = itm.x
        y_i = itm.y
        sum_delx_sqr = zero(Float64)
        sum_dely_sqr = zero(Float64)
        sum_delx_dely = zero(Float64)
        fill!(sum_delx_delq, zero(Float64))
        fill!(sum_dely_delq, zero(Float64))
        for conn in itm.conn
            x_k = globaldata[conn].x
            y_k = globaldata[conn].y
            delx = x_k - x_i
            dely = y_k - y_i
            dist = hypot(delx, dely)
            weights = dist ^ power
            sum_delx_sqr += ((delx * delx) * weights)
            sum_dely_sqr += ((dely * dely) * weights)
            sum_delx_dely += ((delx * dely) * weights)

            @. qi_tilde = itm.q - 0.5 * (delx * itm.dq[1] + dely * itm.dq[2])
            @. qk_tilde = globaldata[conn].q - 0.5 * (delx * globaldata[conn].dq[1] + dely * globaldata[conn].dq[2])

            @. sum_delx_delq += (weights * delx * (qk_tilde - qi_tilde))
            @. sum_dely_delq += (weights * dely * (qk_tilde - qi_tilde))
        end
        det = (sum_delx_sqr * sum_dely_sqr) - (sum_delx_dely * sum_delx_dely)
        one_by_det = 1.0 / det
        @. @views tempdq[idx, 1, :] = one_by_det * (sum_delx_delq * sum_dely_sqr - sum_dely_delq * sum_delx_dely)
        @. @views tempdq[idx, 2, :] = one_by_det * (sum_dely_delq * sum_delx_sqr - sum_delx_delq * sum_delx_dely)
    end

    for (idx, itm) in enumerate(globaldata)
        for iter in 1:4
            itm.dq[1][iter] = tempdq[idx, 1, iter]
            itm.dq[2][iter] = tempdq[idx, 2, iter]
        end
    end
    return nothing
end