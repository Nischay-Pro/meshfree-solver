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

function fpi_solver_cuda(iter, gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, gpuConfigData, gpuSumResSqr, gpuSumResSqrOutput, threadsperblock,blockspergrid, numPoints)

    # dev::CuDevice=CuDevice(0)
    str = CuStream()
    # ctx = CuContext(dev)
    res_old = 0
    println("Blocks per grid is ")
    println(blockspergrid)
    residue_io = open("residue_cuda.txt", "a+")
    # fluxthreadsperblock = Int(max(threadsperblock / 4, 32))
    # fluxblockspergrid = Int(ceil(numPoints/fluxthreadsperblock))

    # gpuGlobalDataCommon = CuArray(globalDataCommon)
    for i in 1:iter
        if i == 1
            println("Compiling CUDA Kernel. This might take a while...")
        end
        @cuda blocks=blockspergrid threads=threadsperblock func_delta_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, gpuConfigData)

        for rk in 1:4
            @cuda blocks=blockspergrid threads=threadsperblock q_var_cuda_kernel(gpuGlobalDataFixedPoint, gpuGlobalDataRest, numPoints)
            # synchronize(str)
            # @cuprintf("\n It is %lf ", gpuGlobalDataCommon[31, 3])
            @cuda blocks=blockspergrid threads=threadsperblock q_var_derivatives_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, gpuConfigData, numPoints)
            # synchronize(str)
            # @cuprintf("\n It is %lf ", gpuGlobalDataCommon[31, 3])
            @cuda blocks= blockspergrid threads= threadsperblock cal_flux_residual_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, gpuConfigData, numPoints)
            # synchronize(str)
            # @cuprintf("\n It is %f ", gpuGlobalDataCommon[31, 3])
            # synchronize(str)
            # @cuprintf("\n It is %lf ", gpuGlobalDataCommon[31, 3])
            @cuda blocks=blockspergrid threads=threadsperblock state_update_kernel(gpuGlobalDataFixedPoint, gpuGlobalDataConn, gpuGlobalDataRest, gpuConfigData, gpuSumResSqr, numPoints, rk)
        end
        synchronize(str)
        gpu_reduced(+, gpuSumResSqr, gpuSumResSqrOutput)
        temp_gpu = Array(gpuSumResSqrOutput)[1]
    # #     # @cuprintf("\n It is ss %lf ", gpuGlobalDataCommon[31, 3])
    # #     # gpu_reduce(+, gpuSumResSqr, gpuSumResSqrOutput)

    #     # temp_gpu = Array(gpuSumResSqrOutput)[1]
        residue = sqrt(temp_gpu) / numPoints
            if i <= 2
                res_old = residue
                residue = 0
            else
                residue = log10(residue / res_old)
            end

        @printf(residue_io, "%d %s\n", i, residue)
        println("Iteration Number ", i, " Residue ", residue)
    end
    synchronize()
    close(residue_io)
    return nothing
end

@inline function q_var_cuda_kernel(gpuGlobalDataFixedPoint, gpuGlobalDataRest, numPoints)
    tx = threadIdx().x
    bx = blockIdx().x - 1
    bw = blockDim().x
    idx = bx * bw + tx
    # itm = CuArray(Float64, 145)
    # if idx == 3
    #     @cuprintf("\n 1 It is %lf ", gpuGlobalDataCommon[31, 3])
    # end
    if idx > 0 && idx <= numPoints

        rho = gpuGlobalDataRest[1, idx]
        u1 = gpuGlobalDataRest[2, idx]
        u2 = gpuGlobalDataRest[3, idx]
        pr = gpuGlobalDataRest[4, idx]
        # gpuGlobalDataCommon[27,idx] = rho
        beta = 0.5 * (rho / pr)
        # @cuprintf beta
        gpuGlobalDataRest[9, idx] = CUDAnative.log(rho) + CUDAnative.log(beta) * 2.5 - beta * ((u1 * u1) + (u2 * u2))
        two_times_beta = 2.0 * beta
        gpuGlobalDataRest[10, idx] = two_times_beta * u1
        gpuGlobalDataRest[11, idx] = two_times_beta * u2
        gpuGlobalDataRest[12, idx] = -two_times_beta
    end
    # if idx ==3
    #     @cuprintf("\n %.17f %.17f %.17f %.17f", gpuGlobalDataRest[9, idx],gpuGlobalDataRest[10, idx],gpuGlobalDataRest[11, idx],gpuGlobalDataRest[12, idx])
    # end
    # sync_threads()
    return nothing
end

function q_var_derivatives_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, gpuConfigData, numPoints)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    # itm = CuArray(Float64, 145)

    if idx > 0 && idx <= numPoints
        sum_delx_sqr = 0.0
        sum_dely_sqr = 0.0
        sum_delx_dely = 0.0
        conn = 0.0
        sum_delx_delq1, sum_delx_delq2, sum_delx_delq3, sum_delx_delq4 = 0.0,0.0,0.0,0.0
        sum_dely_delq1, sum_dely_delq2, sum_dely_delq3, sum_dely_delq4 = 0.0,0.0,0.0,0.0
        x_i = gpuGlobalDataFixedPoint[idx].x
        y_i = gpuGlobalDataFixedPoint[idx].y
        temp = 0.0
        # power = gpuConfigData[6]

        q1, q2, q3, q4 = gpuGlobalDataRest[9, idx], gpuGlobalDataRest[10, idx], gpuGlobalDataRest[11, idx], gpuGlobalDataRest[12, idx]

        gpuGlobalDataRest[21, idx], gpuGlobalDataRest[22, idx], gpuGlobalDataRest[23, idx], gpuGlobalDataRest[24, idx] = gpuGlobalDataRest[9, idx],
			gpuGlobalDataRest[10, idx],gpuGlobalDataRest[11, idx],gpuGlobalDataRest[12, idx]
		gpuGlobalDataRest[25, idx], gpuGlobalDataRest[26, idx], gpuGlobalDataRest[27, idx], gpuGlobalDataRest[28, idx] = gpuGlobalDataRest[9, idx],
			gpuGlobalDataRest[10, idx],gpuGlobalDataRest[11, idx],gpuGlobalDataRest[12, idx]

        for iter in 5:14
            conn = gpuGlobalDataConn[iter, idx]
            if conn == 0.0
                break
            end

            delx = gpuGlobalDataFixedPoint[conn].x - x_i
            dely = gpuGlobalDataFixedPoint[conn].y - y_i
            # dist = CUDAnative.hypot(delx, dely)

            # weights = CUDAnative.pow(dist, gpuConfigData[6])
            weights = 1.0

            sum_delx_sqr = sum_delx_sqr + ((delx * delx) * weights)
            sum_dely_sqr = sum_dely_sqr + ((dely * dely) * weights)
            sum_delx_dely = sum_delx_dely + ((delx * dely) * weights)

            temp = gpuGlobalDataRest[9, conn] - q1
            sum_delx_delq1 += (weights * delx * temp)
            sum_dely_delq1 += (weights * dely * temp)

            temp = gpuGlobalDataRest[10, conn] - q2
            sum_delx_delq2 += (weights * delx * temp)
            sum_dely_delq2 += (weights * dely * temp)

            temp = gpuGlobalDataRest[11, conn] - q3
            sum_delx_delq3 += (weights * delx * temp)
            sum_dely_delq3 += (weights * dely * temp)

            temp = gpuGlobalDataRest[12, conn] - q4
            sum_delx_delq4 += (weights * delx * temp)
            sum_dely_delq4 += (weights * dely * temp)

            update_q(gpuGlobalDataRest, idx, 1, conn)
			update_q(gpuGlobalDataRest, idx, 2, conn)
			update_q(gpuGlobalDataRest, idx, 3, conn)
			update_q(gpuGlobalDataRest, idx, 4, conn)
        end
        det = (sum_delx_sqr * sum_dely_sqr) - (sum_delx_dely * sum_delx_dely)
        one_by_det = 1.0 / det
        gpuGlobalDataRest[13, idx] = one_by_det * (sum_delx_delq1 * sum_dely_sqr - sum_dely_delq1 * sum_delx_dely)
        gpuGlobalDataRest[14, idx] = one_by_det * (sum_delx_delq2 * sum_dely_sqr - sum_dely_delq2 * sum_delx_dely)
        gpuGlobalDataRest[15, idx] = one_by_det * (sum_delx_delq3 * sum_dely_sqr - sum_dely_delq3 * sum_delx_dely)
        gpuGlobalDataRest[16, idx] = one_by_det * (sum_delx_delq4 * sum_dely_sqr - sum_dely_delq4 * sum_delx_dely)
        gpuGlobalDataRest[17, idx] = one_by_det * (sum_dely_delq1 * sum_delx_sqr - sum_delx_delq1 * sum_delx_dely)
        gpuGlobalDataRest[18, idx] = one_by_det * (sum_dely_delq2 * sum_delx_sqr - sum_delx_delq2 * sum_delx_dely)
        gpuGlobalDataRest[19, idx] = one_by_det * (sum_dely_delq3 * sum_delx_sqr - sum_delx_delq3 * sum_delx_dely)
        gpuGlobalDataRest[20, idx] = one_by_det * (sum_dely_delq4 * sum_delx_sqr - sum_delx_delq4 * sum_delx_dely)
        # @cuda dynamic=true threads=4 max_min_kernel(gpuGlobalDataCommon, idx)
        # CUDAnative.synchronize()
    end
    #q_var_derivatives_innerloops_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, numPoints)
    #q_var_derivatives_innerloops_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, numPoints)
    # sync_threads()
    return nothing
end

function q_var_derivatives_innerloops_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, numPoints)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    # itm = CuArray(Float64, 145)

    if idx > 0 && idx <= numPoints
        sum_delx_sqr = 0.0
        sum_dely_sqr = 0.0
        sum_delx_dely = 0.0
        conn = 0.0
        sum_delx_delq1, sum_delx_delq2, sum_delx_delq3, sum_delx_delq4 = 0.0,0.0,0.0,0.0
        sum_dely_delq1, sum_dely_delq2, sum_dely_delq3, sum_dely_delq4 = 0.0,0.0,0.0,0.0
        x_i = gpuGlobalDataFixedPoint[idx].x
        y_i = gpuGlobalDataFixedPoint[idx].y
        temp1 = 0.0
        temp2 = 0.0
        # power = gpuConfigData[6]

        q1, q2, q3, q4 = gpuGlobalDataRest[9, idx], gpuGlobalDataRest[10, idx], gpuGlobalDataRest[11, idx], gpuGlobalDataRest[12, idx]

        for iter in 5:14
            conn = gpuGlobalDataConn[iter, idx]
            if conn == 0.0
                break
            end

            delx = gpuGlobalDataFixedPoint[conn].x - x_i
            dely = gpuGlobalDataFixedPoint[conn].y - y_i
            # dist = CUDAnative.hypot(delx, dely)
            # weights = CUDAnative.pow(dist, gpuConfigData[6])
            weights = 1.0

            sum_delx_sqr = sum_delx_sqr + ((delx * delx) * weights)
            sum_dely_sqr = sum_dely_sqr + ((dely * dely) * weights)
            sum_delx_dely = sum_delx_dely + ((delx * dely) * weights)

            temp1 = q1 - 0.5 * (delx * gpuGlobalDataRest[13, idx] + dely * gpuGlobalDataRest[17, idx])
            temp2 = gpuGlobalDataRest[9, conn] - 0.5 * (delx * gpuGlobalDataRest[13, conn] + dely * gpuGlobalDataRest[17, conn])
            sum_delx_delq1 += (weights * delx * (temp2 - temp1))
            sum_dely_delq1 += (weights * dely * (temp2 - temp1))

            temp1 = q2 - 0.5 * (delx * gpuGlobalDataRest[14, idx] + dely * gpuGlobalDataRest[18, idx])
            temp2 = gpuGlobalDataRest[10, conn] - 0.5 * (delx * gpuGlobalDataRest[14, conn] + dely * gpuGlobalDataRest[18, conn])
            sum_delx_delq2 += (weights * delx * (temp2 - temp1))
            sum_dely_delq2 += (weights * dely * (temp2 - temp1))

            temp1 = q3 - 0.5 * (delx * gpuGlobalDataRest[15, idx] + dely * gpuGlobalDataRest[19, idx])
            temp2 = gpuGlobalDataRest[11, conn] - 0.5 * (delx * gpuGlobalDataRest[15, conn] + dely * gpuGlobalDataRest[19, conn])
            sum_delx_delq3 += (weights * delx * (temp2 - temp1))
            sum_dely_delq3 += (weights * dely * (temp2 - temp1))

            temp1 = q4 - 0.5 * (delx * gpuGlobalDataRest[16, idx] + dely * gpuGlobalDataRest[20, idx])
            temp2 = gpuGlobalDataRest[12, conn] - 0.5 * (delx * gpuGlobalDataRest[16, conn] + dely * gpuGlobalDataRest[20, conn])
            sum_delx_delq4 += (weights * delx * (temp2 - temp1))
            sum_dely_delq4 += (weights * dely * (temp2 - temp1))

        end
        det = (sum_delx_sqr * sum_dely_sqr) - (sum_delx_dely * sum_delx_dely)
        one_by_det = 1.0 / det
        gpuGlobalDataRest[13, idx] = one_by_det * (sum_delx_delq1 * sum_dely_sqr - sum_dely_delq1 * sum_delx_dely)
        gpuGlobalDataRest[14, idx] = one_by_det * (sum_delx_delq2 * sum_dely_sqr - sum_dely_delq2 * sum_delx_dely)
        gpuGlobalDataRest[15, idx] = one_by_det * (sum_delx_delq3 * sum_dely_sqr - sum_dely_delq3 * sum_delx_dely)
        gpuGlobalDataRest[16, idx] = one_by_det * (sum_delx_delq4 * sum_dely_sqr - sum_dely_delq4 * sum_delx_dely)
        gpuGlobalDataRest[17, idx] = one_by_det * (sum_dely_delq1 * sum_delx_sqr - sum_delx_delq1 * sum_delx_dely)
        gpuGlobalDataRest[18, idx] = one_by_det * (sum_dely_delq2 * sum_delx_sqr - sum_delx_delq2 * sum_delx_dely)
        gpuGlobalDataRest[19, idx] = one_by_det * (sum_dely_delq3 * sum_delx_sqr - sum_delx_delq3 * sum_delx_dely)
        gpuGlobalDataRest[20, idx] = one_by_det * (sum_dely_delq4 * sum_delx_sqr - sum_delx_delq4 * sum_delx_dely)
        # @cuda dynamic=true threads=4 max_min_kernel(gpuGlobalDataCommon, idx)
        # CUDAnative.synchronize()
    end
    # sync_threads()
    return nothing
end

@inline function update_q(gpuGlobalDataRest, idx, i, conn)
    if gpuGlobalDataRest[20+i, idx] < gpuGlobalDataRest[8+i, conn]
        gpuGlobalDataRest[20+i, idx] = gpuGlobalDataRest[8+i, conn]
    end
    if gpuGlobalDataRest[24+i, idx] > gpuGlobalDataRest[8+i, conn]
        gpuGlobalDataRest[24+i, idx] = gpuGlobalDataRest[8+i, conn]
    end
    return nothing
end


@inline function reduce_warp(op::F, val::T)::T where {F<:Function,T}
    offset = CUDAnative.warpsize() ÷ 2
    # TODO: this can be unrolled if warpsize is known...
    while offset > 0
        val = op(val, shfl_down(val, offset))
        offset ÷= 2
    end
    return val
end

# Reduce a value across a block, using shared memory for communication
@inline function reduce_block(op::F, val::T)::T where {F<:Function,T}
    # shared mem for 32 partial sums
    shared = @cuStaticSharedMem(T, 32)

    wid, lane = fldmod1(threadIdx().x, CUDAnative.warpsize())

    # each warp performs partial reduction
    val = reduce_warp(op, val)

    # write reduced value to shared memory
    if lane == 1
        @inbounds shared[wid] = val
    end

    # wait for all partial reductions
    sync_threads()

    # read from shared memory only if that warp existed
    @inbounds val = (threadIdx().x <= fld(blockDim().x, CUDAnative.warpsize())) ? shared[lane] : zero(T)

    # final reduce within first warp
    if wid == 1
        val = reduce_warp(op, val)
    end

    return val
end

# Reduce an array across a complete grid
@inline function reduce_grid(op::F, input::CuDeviceVector{T}, output::CuDeviceVector{T},
                     len::Integer) where {F<:Function,T}
    # TODO: neutral element depends on the operator (see Base's 2 and 3 argument `reduce`)
    val = zero(T)

    # reduce multiple elements per thread (grid-stride loop)
    # TODO: step range (see JuliaGPU/CUDAnative.jl#12)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    step = blockDim().x * gridDim().x
    while i <= len
        @inbounds val = op(val, input[i])
        i += step
    end

    val = reduce_block(op, val)

    if threadIdx().x == 1
        @inbounds output[blockIdx().x] = val
    end

    return
end

"""
Reduce a large array.
Kepler-specific implementation, ie. you need sm_30 or higher to run this code.
"""
function gpu_reduced(op::Function, input::CuArray{T}, output::CuArray{T}) where {T}
    len = length(input)

    # TODO: these values are hardware-dependent, with recent GPUs supporting more threads
    threads = 512
    blocks = min((len + threads - 1) ÷ threads, 1024)

    # the output array must have a size equal to or larger than the number of thread blocks
    # in the grid because each block writes to a unique location within the array.
    if length(output) < blocks
        throw(ArgumentError("output array too small, should be at least $blocks elements"))
    end

    @cuda blocks=blocks threads=threads reduce_grid(op, input, output, len)
    @cuda threads=1024 reduce_grid(op, output, output, blocks)
end
