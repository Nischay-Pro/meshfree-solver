function getInitialPrimitive(configData)
    rho_inf::Float64 = configData["core"]["rho_inf"]
    mach::Float64 = configData["core"]["mach"]
    machcos = mach * cos(calculateTheta(configData))
    machsin = mach * sin(calculateTheta(configData))
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

function calculateConnectivity(globaldata, idx, configData)
    ptInterest = globaldata[idx]
    currx = ptInterest.x
    curry = ptInterest.y
    nx = ptInterest.nx
    ny = ptInterest.ny

    flag = ptInterest.flag_1

    xpos_conn,xneg_conn,ypos_conn,yneg_conn = Array{Int,1}(undef, 0),Array{Int,1}(undef, 0),Array{Int,1}(undef, 0),Array{Int,1}(undef, 0)

    tx = ny
    ty = -nx

    interior = configData["point"]["interior"]
    wall = configData["point"]["wall"]
    outer = configData["point"]["outer"]

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
        if flag == interior
            if deln <= 0.0
                push!(ypos_conn, itm)
            end
            if deln >= 0.0
                push!(yneg_conn, itm)
            end
        elseif flag == wall
            push!(yneg_conn, itm)
        elseif flag == outer
            push!(ypos_conn, itm)
        end
    end
    return (xpos_conn, xneg_conn, ypos_conn, yneg_conn)
end

function getPointDetails(gdata, p_i)
    tx = threadIdx().x
    bx = blockIdx().x - 1
    bw = blockDim().x
    idx = bx * bw + tx
    # if idx == p_i
    # end
    return nothing
end

function fpi_solver_cuda(iter, inner, gpuGlobalDataConn, gpuGlobalDataConnSection, gpuGlobalDataFauxFixed, gpuGlobalDataRest, gpuConfigData, gpuSumResSqr, gpuSumResSqrOutput, threadsperblock, blockspergrid, numPoints)

    # dev::CuDevice=CuDevice(0)
    str = CuStream()
    # ctx = CuContext(dev)
    res_old = 0
    println("Blocks per grid is ")
    println(blockspergrid)
    # residue_io = open("residue_cuda.txt", "a+")
    # fluxthreadsperblock = Int(max(threadsperblock / 4, 32))
    # fluxblockspergrid = Int(ceil(numPoints/fluxthreadsperblock))

    for i in 1:iter
        if i == 1
            println("Compiling CUDA Kernel. This might take a while...")
        end
        @cuda blocks=blockspergrid threads=threadsperblock func_delta_kernel(gpuGlobalDataConn, gpuGlobalDataFauxFixed, gpuGlobalDataRest, gpuConfigData, numPoints)
        # @cuda blocks=blockspergrid threads=threadsperblock getPointDetails(gpuGlobalDataRest, 2000)
        for rk in 1:4
            # @cuprintf("\n Value is %f", gpuGlobalDataRest[3, 31])
            @cuda blocks=blockspergrid threads=threadsperblock q_var_cuda_kernel(gpuGlobalDataRest, numPoints)
            # synchronize(str)
            # @cuprintf("\n It is %lf ", gpuGlobalDataCommon[31, 3])
            @cuda blocks=blockspergrid threads=threadsperblock q_var_derivatives_kernel(gpuGlobalDataConn,  gpuGlobalDataFauxFixed, gpuGlobalDataRest, gpuConfigData, numPoints)
            innermost = 0
            while innermost < inner
                @cuda blocks= blockspergrid threads= threadsperblock q_var_derivatives_innerloops_kernel(gpuGlobalDataConn, gpuGlobalDataFauxFixed, gpuGlobalDataRest, gpuConfigData, numPoints)
                @cuda blocks= blockspergrid threads= threadsperblock update_innerloop_kernel(gpuGlobalDataRest, numPoints)
                innermost += 1
            end
            # @cuda blocks=blockspergrid threads=threadsperblock getPointDetails(gpuGlobalDataRest, 2000)
            # synchronize(str)
            # @cuprintf("\n It is %lf ", gpuGlobalDataCommon[31, 3])
            @cuda blocks= blockspergrid threads= threadsperblock cal_flux_residual_kernel(gpuGlobalDataConn, gpuGlobalDataConnSection, gpuGlobalDataFauxFixed, gpuGlobalDataRest, gpuConfigData, numPoints)
            # @cuda blocks=blockspergrid threads=threadsperblock getPointDetails(gpuGlobalDataRest, 3)
            # synchronize(str)
            # @cuprintf("\n It is %f ", gpuGlobalDataCommon[31, 3])
            # synchronize(str)
            # @cuprintf("\n It is %lf ", gpuGlobalDataCommon[31, 3])
            @cuda blocks=blockspergrid threads=threadsperblock state_update_kernel(gpuGlobalDataFauxFixed, gpuGlobalDataConn, gpuGlobalDataRest, gpuConfigData, gpuSumResSqr, numPoints, rk)
            # @cuda blocks=blockspergrid threads=threadsperblock getPointDetails(gpuGlobalDataRest, 2000)
        end
        synchronize(str)
        temp_gpu = reduce(+, gpuSumResSqr)
        res_new = temp_gpu
        residue = 0.0
        if i <= 2
            res_old = res_new
            residue = 0
        else
            residue = log10(sqrt(sum(res_new))/sqrt(sum(res_old)))
        end

        # @printf(residue_io, "%d %.17f\n", i, residue)
        println(IOContext(stdout, :compact => false),"Iteration Number ", i, " Residue ", residue)
    end
    synchronize()
    # close(residue_io)
    return nothing
end

function q_var_cuda_kernel(gpuGlobalDataRest, numPoints)
    tx = threadIdx().x
    bx = blockIdx().x - 1
    bw = blockDim().x
    idx = bx * bw + tx
    # itm = CuArray(Float64, 145)

    if idx > 0 && idx <= numPoints

        # if idx == 2000
        #     # @cuprintf("\n %.17f %.17f ==========================================", gpuGlobalDataFixedPoint[idx].x ,gpuGlobalDataFixedPoint[idx].y)
        #     @cuprintf("\n ==========================================")
        # end
        rho = gpuGlobalDataRest[idx, 1]
        u1 = gpuGlobalDataRest[idx, 2]
        u2 = gpuGlobalDataRest[idx, 3]
        pr = gpuGlobalDataRest[idx, 4]
        # gpuGlobalDataCommon[27,idx] = rho
        beta = 0.5 * (rho / pr)
        # @cuprintf beta
        gpuGlobalDataRest[idx, 9] = CUDA.log(rho) + CUDA.log(beta) * 2.5 - beta * ((u1 * u1) + (u2 * u2))
        two_times_beta = 2.0 * beta
        gpuGlobalDataRest[idx, 10] = two_times_beta * u1
        gpuGlobalDataRest[idx, 11] = two_times_beta * u2
        gpuGlobalDataRest[idx, 12] = -two_times_beta
        # if idx == 3
            # @cuprintln("\n Value is ", gpuGlobalDataRest[3, 31])
        # end
        q1, q2, q3, q4 = gpuGlobalDataRest[idx, 9], gpuGlobalDataRest[idx, 10], gpuGlobalDataRest[idx, 11], gpuGlobalDataRest[idx, 12]
        gpuGlobalDataRest[idx, 21], gpuGlobalDataRest[idx, 22], gpuGlobalDataRest[idx, 23], gpuGlobalDataRest[idx, 24] = q1,q2,q3,q4
		gpuGlobalDataRest[idx, 25], gpuGlobalDataRest[idx, 26], gpuGlobalDataRest[idx, 27], gpuGlobalDataRest[idx, 28] = q1,q2,q3,q4
    end

    # sync_threads()
    return nothing
end

function q_var_derivatives_kernel(gpuGlobalDataConn, gpuGlobalDataFauxFixed, gpuGlobalDataRest, gpuConfigData, numPoints)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    # itm = CuArray(Float64, 145)

    if idx > 0 && idx <= numPoints
        x_i = gpuGlobalDataFauxFixed[idx]
        y_i = gpuGlobalDataFauxFixed[idx + numPoints]

        sum_delx_sqr = 0.0
        sum_dely_sqr = 0.0
        sum_delx_dely = 0.0
        conn = 0.0
        sum_delx_delq1, sum_delx_delq2, sum_delx_delq3, sum_delx_delq4 = 0.0,0.0,0.0,0.0
        sum_dely_delq1, sum_dely_delq2, sum_dely_delq3, sum_dely_delq4 = 0.0,0.0,0.0,0.0
        q1, q2, q3, q4 = gpuGlobalDataRest[idx, 9], gpuGlobalDataRest[idx, 10], gpuGlobalDataRest[idx, 11], gpuGlobalDataRest[idx, 12]
        temp = 0.0
        power = gpuConfigData[6]

        for iter in 1:20
            conn = gpuGlobalDataConn[idx, iter]
            if conn == 0
                break
            end

            delx = gpuGlobalDataFauxFixed[conn] - x_i
            dely = gpuGlobalDataFauxFixed[conn + numPoints] - y_i
            dist = CUDA.hypot(delx, dely)

            weights = CUDA.pow(dist, power)
            # weights = 1.0

            sum_delx_sqr = sum_delx_sqr + ((delx * delx) * weights)
            sum_dely_sqr = sum_dely_sqr + ((dely * dely) * weights)
            sum_delx_dely = sum_delx_dely + ((delx * dely) * weights)

            temp = gpuGlobalDataRest[conn, 9] - q1
            update_q(gpuGlobalDataRest, idx, 1, gpuGlobalDataRest[conn, 9])
            sum_delx_delq1 += (weights * delx * temp)
            sum_dely_delq1 += (weights * dely * temp)

            temp = gpuGlobalDataRest[conn, 10] - q2
            update_q(gpuGlobalDataRest, idx, 2, gpuGlobalDataRest[conn, 10])
            sum_delx_delq2 += (weights * delx * temp)
            sum_dely_delq2 += (weights * dely * temp)

            temp = gpuGlobalDataRest[conn, 11] - q3
            update_q(gpuGlobalDataRest, idx, 3, gpuGlobalDataRest[conn, 11])
            sum_delx_delq3 += (weights * delx * temp)
            sum_dely_delq3 += (weights * dely * temp)

            temp = gpuGlobalDataRest[conn, 12] - q4
            update_q(gpuGlobalDataRest, idx, 4, gpuGlobalDataRest[conn, 12])
            sum_delx_delq4 += (weights * delx * temp)
            sum_dely_delq4 += (weights * dely * temp)

        end
        det = (sum_delx_sqr * sum_dely_sqr) - (sum_delx_dely * sum_delx_dely)
        one_by_det = 1.0 / det
        gpuGlobalDataRest[idx, 13] = one_by_det * (sum_delx_delq1 * sum_dely_sqr - sum_dely_delq1 * sum_delx_dely)
        gpuGlobalDataRest[idx, 14] = one_by_det * (sum_delx_delq2 * sum_dely_sqr - sum_dely_delq2 * sum_delx_dely)
        gpuGlobalDataRest[idx, 15] = one_by_det * (sum_delx_delq3 * sum_dely_sqr - sum_dely_delq3 * sum_delx_dely)
        gpuGlobalDataRest[idx, 16] = one_by_det * (sum_delx_delq4 * sum_dely_sqr - sum_dely_delq4 * sum_delx_dely)
        gpuGlobalDataRest[idx, 17] = one_by_det * (sum_dely_delq1 * sum_delx_sqr - sum_delx_delq1 * sum_delx_dely)
        gpuGlobalDataRest[idx, 18] = one_by_det * (sum_dely_delq2 * sum_delx_sqr - sum_delx_delq2 * sum_delx_dely)
        gpuGlobalDataRest[idx, 19] = one_by_det * (sum_dely_delq3 * sum_delx_sqr - sum_delx_delq3 * sum_delx_dely)
        gpuGlobalDataRest[idx, 20] = one_by_det * (sum_dely_delq4 * sum_delx_sqr - sum_delx_delq4 * sum_delx_dely)
        # @cuda dynamic=true threads=4 max_min_kernel(gpuGlobalDataCommon, idx)
        # CUDA.synchronize()
    end
    return nothing
end

function q_var_derivatives_innerloops_kernel(gpuGlobalDataConn, gpuGlobalDataFauxFixed, gpuGlobalDataRest, gpuConfigData, numPoints)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    # itm = CuArray(Float64, 145)

    if idx > 0 && idx <= numPoints
        sum_delx_sqr = 0.0
        sum_dely_sqr = 0.0
        sum_delx_dely = 0.0
        conn = 0.0
        sum_delx_delq1, sum_delx_delq2, sum_delx_delq3, sum_delx_delq4 = 0.0,0.0,0.0,0.0
        sum_dely_delq1, sum_dely_delq2, sum_dely_delq3, sum_dely_delq4 = 0.0,0.0,0.0,0.0
        x_i = gpuGlobalDataFauxFixed[idx]
        y_i = gpuGlobalDataFauxFixed[idx + numPoints]
        temp1 = 0.0
        temp2 = 0.0
        power = gpuConfigData[6]

        q1, q2, q3, q4 = gpuGlobalDataRest[idx, 9], gpuGlobalDataRest[idx, 10], gpuGlobalDataRest[idx, 11], gpuGlobalDataRest[idx, 12]

        for iter in 1:20
            conn = gpuGlobalDataConn[idx, iter]
            if conn == 0
                break
            end

            delx = gpuGlobalDataFauxFixed[conn] - x_i
            dely = gpuGlobalDataFauxFixed[conn + numPoints] - y_i
            dist = CUDA.hypot(delx, dely)
            weights = CUDA.pow(dist, power)
            # weights = 1.0

            sum_delx_sqr = sum_delx_sqr + ((delx * delx) * weights)
            sum_dely_sqr = sum_dely_sqr + ((dely * dely) * weights)
            sum_delx_dely = sum_delx_dely + ((delx * dely) * weights)

            temp1 = q1 - 0.5 * (delx * gpuGlobalDataRest[idx, 13] + dely * gpuGlobalDataRest[idx, 17])
            temp2 = gpuGlobalDataRest[conn, 9] - 0.5 * (delx * gpuGlobalDataRest[conn, 13] + dely * gpuGlobalDataRest[conn, 17])
            sum_delx_delq1 += (weights * delx * (temp2 - temp1))
            sum_dely_delq1 += (weights * dely * (temp2 - temp1))

            temp1 = q2 - 0.5 * (delx * gpuGlobalDataRest[idx, 14] + dely * gpuGlobalDataRest[idx, 18])
            temp2 = gpuGlobalDataRest[conn, 10] - 0.5 * (delx * gpuGlobalDataRest[conn, 14] + dely * gpuGlobalDataRest[conn, 18])
            sum_delx_delq2 += (weights * delx * (temp2 - temp1))
            sum_dely_delq2 += (weights * dely * (temp2 - temp1))

            temp1 = q3 - 0.5 * (delx * gpuGlobalDataRest[idx, 15] + dely * gpuGlobalDataRest[idx, 19])
            temp2 = gpuGlobalDataRest[conn, 11] - 0.5 * (delx * gpuGlobalDataRest[conn, 15] + dely * gpuGlobalDataRest[conn, 19])
            sum_delx_delq3 += (weights * delx * (temp2 - temp1))
            sum_dely_delq3 += (weights * dely * (temp2 - temp1))

            temp1 = q4 - 0.5 * (delx * gpuGlobalDataRest[idx, 16] + dely * gpuGlobalDataRest[idx, 20])
            temp2 = gpuGlobalDataRest[conn, 12] - 0.5 * (delx * gpuGlobalDataRest[conn, 16] + dely * gpuGlobalDataRest[conn, 20])
            sum_delx_delq4 += (weights * delx * (temp2 - temp1))
            sum_dely_delq4 += (weights * dely * (temp2 - temp1))

        end
        det = (sum_delx_sqr * sum_dely_sqr) - (sum_delx_dely * sum_delx_dely)
        one_by_det = 1.0 / det
        gpuGlobalDataRest[idx, 34] = one_by_det * (sum_delx_delq1 * sum_dely_sqr - sum_dely_delq1 * sum_delx_dely)
        gpuGlobalDataRest[idx, 35] = one_by_det * (sum_delx_delq2 * sum_dely_sqr - sum_dely_delq2 * sum_delx_dely)
        gpuGlobalDataRest[idx, 36] = one_by_det * (sum_delx_delq3 * sum_dely_sqr - sum_dely_delq3 * sum_delx_dely)
        gpuGlobalDataRest[idx, 37] = one_by_det * (sum_delx_delq4 * sum_dely_sqr - sum_dely_delq4 * sum_delx_dely)
        gpuGlobalDataRest[idx, 5] = one_by_det * (sum_dely_delq1 * sum_delx_sqr - sum_delx_delq1 * sum_delx_dely)
        gpuGlobalDataRest[idx, 6] = one_by_det * (sum_dely_delq2 * sum_delx_sqr - sum_delx_delq2 * sum_delx_dely)
        gpuGlobalDataRest[idx, 7] = one_by_det * (sum_dely_delq3 * sum_delx_sqr - sum_delx_delq3 * sum_delx_dely)
        gpuGlobalDataRest[idx, 8] = one_by_det * (sum_dely_delq4 * sum_delx_sqr - sum_delx_delq4 * sum_delx_dely)
        # @cuda dynamic=true threads=4 max_min_kernel(gpuGlobalDataCommon, idx)
        # CUDA.synchronize()
    end
    # sync_threads()
    return nothing
end

function update_innerloop_kernel(gpuGlobalDataRest, numPoints)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    # itm = CuArray(Float64, 145)

    if idx > 0 && idx <= numPoints
        gpuGlobalDataRest[idx, 13] = gpuGlobalDataRest[idx, 34]
        gpuGlobalDataRest[idx, 14] = gpuGlobalDataRest[idx, 35]
        gpuGlobalDataRest[idx, 15] = gpuGlobalDataRest[idx, 36]
        gpuGlobalDataRest[idx, 16] = gpuGlobalDataRest[idx, 37]
        gpuGlobalDataRest[idx, 17] = gpuGlobalDataRest[idx, 5]
        gpuGlobalDataRest[idx, 18] = gpuGlobalDataRest[idx, 6]
        gpuGlobalDataRest[idx, 19] = gpuGlobalDataRest[idx, 7]
        gpuGlobalDataRest[idx, 20] = gpuGlobalDataRest[idx, 8]
    end
    return nothing
end

function update_q(gpuGlobalDataRest, idx, i, conn_store)
    if gpuGlobalDataRest[idx, 20+i] < conn_store
        gpuGlobalDataRest[idx, 20+i] = conn_store
    end
    if gpuGlobalDataRest[idx, 24+i] > conn_store
        gpuGlobalDataRest[idx, 24+i] = conn_store
    end
    return nothing
end
