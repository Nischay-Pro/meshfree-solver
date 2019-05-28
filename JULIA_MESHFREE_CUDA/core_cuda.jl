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

    xpos_conn,xneg_conn,ypos_conn,yneg_conn = Array{Int,1}(undef, 0),Array{Int,1}(undef, 0),Array{Int,1}(undef, 0),Array{Int,1}(undef, 0)

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
        if flag == 2
            if deln <= 0.0
                push!(ypos_conn, itm)
            end
            if deln >= 0.0
                push!(yneg_conn, itm)
            end
        elseif flag == 1
            push!(yneg_conn, itm)
        elseif flag == 3
            push!(ypos_conn, itm)
        end
    end
    return (xpos_conn, xneg_conn, ypos_conn, yneg_conn)
end

# function fpi_solver(iter, globaldata, configData, wallindices, outerindices, interiorindices, res_old)
#     # println(IOContext(stdout, :compact => false), globaldata[1].prim)
#     # print(" 111\n")

#     q_var_derivatives(globaldata, configData)

#     cal_flux_residual(globaldata, wallindices, outerindices, interiorindices, configData)

#     func_delta(globaldata, configData)

#     state_update(globaldata, wallindices, outerindices, interiorindices, configData, iter, res_old)
#     # println("")
#     println(globaldata[3])
# end

function fpi_solver_cuda(iter, gpuGlobalDataCommon, gpuConfigData, threadsperblock,blockspergrid)

    # dev::CuDevice=CuDevice(0)
    str = CuStream()
    # ctx = CuContext(dev)

    println("Blocks per grid is ")
    println(blockspergrid)
    # gpuGlobalDataCommon = CuArray(globalDataCommon)
    for i in 1:iter
        if i == 1
            println("Compiling CUDA Kernel. This might take a while...")
        end

        @cuda blocks=blockspergrid threads=threadsperblock q_var_cuda_kernel(gpuGlobalDataCommon) #, out1, out2)
        synchronize(str)
        # @cuprintf("\n It is %lf ", gpuGlobalDataCommon[31, 3])
        @cuda blocks=blockspergrid threads=threadsperblock q_var_derivatives_kernel(gpuGlobalDataCommon, gpuConfigData)
        synchronize(str)
        # @cuprintf("\n It is %lf ", gpuGlobalDataCommon[31, 3])
        @cuda blocks=blockspergrid threads=threadsperblock cal_flux_residual_kernel(gpuGlobalDataCommon, gpuConfigData)
        synchronize(str)
        # @cuprintf("\n It is %f ", gpuGlobalDataCommon[31, 3])
        @cuda blocks=blockspergrid threads=threadsperblock func_delta_kernel(gpuGlobalDataCommon, gpuConfigData)
        synchronize(str)
        # @cuprintf("\n It is %lf ", gpuGlobalDataCommon[31, 3])
        @cuda blocks=blockspergrid threads=threadsperblock state_update_kernel(gpuGlobalDataCommon, gpuConfigData)
        synchronize(str)
        # @cuprintf("\n It is ss %lf ", gpuGlobalDataCommon[31, 3])
        println("Iteration Number ", i)
    end
    # synchronize()
    # println(Array(out1))
    # @cuprintf("\n It is ss2 %lf ", gpuGlobalDataCommon[31, 3])
    # globalDataCommon = Array(gpuGlobalDataCommon)
    # println("Test is ", globalDataCommon[31,3])
    return nothing
end

function q_var_cuda_kernel(gpuGlobalDataCommon) #out1, out2)
    tx = threadIdx().x
    bx = blockIdx().x - 1
    bw = blockDim().x
    idx = bx * bw + tx
    # itm = CuArray(Float64, 145)
    # if idx == 3
    #     @cuprintf("\n 1 It is %lf ", gpuGlobalDataCommon[31, 3])
    # end
    if idx > 0 && idx <= gpuGlobalDataCommon[1,end]

        rho = gpuGlobalDataCommon[31, idx]
        u1 = gpuGlobalDataCommon[32, idx]
        u2 = gpuGlobalDataCommon[33, idx]
        pr = gpuGlobalDataCommon[34, idx]
        # gpuGlobalDataCommon[27,idx] = rho
        beta = 0.5 * (rho / pr)
        # @cuprintf beta
        gpuGlobalDataCommon[39, idx] = CUDAnative.log(rho) + CUDAnative.log(beta) * 2.5 - beta * ((u1 * u1) + (u2 * u2))
        two_times_beta = 2.0 * beta
        gpuGlobalDataCommon[40, idx] = two_times_beta * u1
        gpuGlobalDataCommon[41, idx] = two_times_beta * u2
        gpuGlobalDataCommon[42, idx] = -two_times_beta
    end
    # if idx ==3
    #     @cuprintf("\n %.17f %.17f %.17f %.17f", gpuGlobalDataCommon[39, idx],gpuGlobalDataCommon[40, idx],gpuGlobalDataCommon[41, idx],gpuGlobalDataCommon[42, idx])
    # end
    sync_threads()
    return nothing
end

function q_var_derivatives_kernel(gpuGlobalDataCommon, gpuConfigData)
    tx = threadIdx().x
    bx = blockIdx().x - 1
    bw = blockDim().x
    idx = bx * bw + tx
    # itm = CuArray(Float64, 145)

    sum_delx_sqr = 0.0
    sum_dely_sqr = 0.0
    sum_delx_dely = 0.0
    sum_delx_delq1, sum_delx_delq2,sum_delx_delq3,sum_delx_delq4 = 0.0,0.0,0.0,0.0
    sum_dely_delq1, sum_dely_delq2,sum_dely_delq3,sum_dely_delq4 = 0.0,0.0,0.0,0.0

    if idx > 0 && idx <= gpuGlobalDataCommon[1,end]
        x_i = gpuGlobalDataCommon[2, idx]
        y_i = gpuGlobalDataCommon[3, idx]
        for iter in 9:28
            conn = Int(gpuGlobalDataCommon[iter, idx])
            if conn == 0.0
                break
            end
            x_k = gpuGlobalDataCommon[2, conn]
            y_k = gpuGlobalDataCommon[3, conn]
            delx = x_k - x_i
            dely = y_k - y_i
            dist = CUDAnative.hypot(delx, dely)
            power = gpuConfigData[6]
            weights = CUDAnative.pow(dist, power)
            sum_delx_sqr = sum_delx_sqr + ((delx * delx) * weights)
            sum_dely_sqr = sum_dely_sqr + ((dely * dely) * weights)
            sum_delx_dely = sum_delx_dely + ((delx * dely) * weights)
            sum_delx_delq1 += (weights * delx * (gpuGlobalDataCommon[39,conn] - gpuGlobalDataCommon[39,idx]))
            sum_dely_delq1 += (weights * dely * (gpuGlobalDataCommon[39,conn] - gpuGlobalDataCommon[39,idx]))
            sum_delx_delq2 += (weights * delx * (gpuGlobalDataCommon[40,conn] - gpuGlobalDataCommon[40,idx]))
            sum_dely_delq2 += (weights * dely * (gpuGlobalDataCommon[40,conn] - gpuGlobalDataCommon[40,idx]))
            sum_delx_delq3 += (weights * delx * (gpuGlobalDataCommon[41,conn] - gpuGlobalDataCommon[41,idx]))
            sum_dely_delq3 += (weights * dely * (gpuGlobalDataCommon[41,conn] - gpuGlobalDataCommon[41,idx]))
            sum_delx_delq4 += (weights * delx * (gpuGlobalDataCommon[42,conn] - gpuGlobalDataCommon[42,idx]))
            sum_dely_delq4 += (weights * dely * (gpuGlobalDataCommon[42,conn] - gpuGlobalDataCommon[42,idx]))
        end
        det = (sum_delx_sqr * sum_dely_sqr) - (sum_delx_dely * sum_delx_dely)
        one_by_det = 1.0 / det
        gpuGlobalDataCommon[43, idx] = one_by_det * (sum_delx_delq1 * sum_dely_sqr - sum_dely_delq1 * sum_delx_dely)
        gpuGlobalDataCommon[44, idx] = one_by_det * (sum_delx_delq2 * sum_dely_sqr - sum_dely_delq2 * sum_delx_dely)
        gpuGlobalDataCommon[45, idx] = one_by_det * (sum_delx_delq3 * sum_dely_sqr - sum_dely_delq3 * sum_delx_dely)
        gpuGlobalDataCommon[46, idx] = one_by_det * (sum_delx_delq4 * sum_dely_sqr - sum_dely_delq4 * sum_delx_dely)
        gpuGlobalDataCommon[47, idx] = one_by_det * (sum_dely_delq1 * sum_delx_sqr - sum_delx_delq1 * sum_delx_dely)
        gpuGlobalDataCommon[48, idx] = one_by_det * (sum_dely_delq2 * sum_delx_sqr - sum_delx_delq2 * sum_delx_dely)
        gpuGlobalDataCommon[49, idx] = one_by_det * (sum_dely_delq3 * sum_delx_sqr - sum_delx_delq3 * sum_delx_dely)
        gpuGlobalDataCommon[50, idx] = one_by_det * (sum_dely_delq4 * sum_delx_sqr - sum_delx_delq4 * sum_delx_dely)
        @cuda dynamic=true threads=4 max_min_kernel(gpuGlobalDataCommon, idx)
        # CUDAnative.synchronize()
    end
    sync_threads()
    return nothing
end

@inline function max_min_kernel(gpuGlobalDataCommon, idx::Int64)
    tx = threadIdx().x
    bx = blockIdx().x - 1
    bw = blockDim().x
    i = bx * bw + tx
    gpuGlobalDataCommon[137+i, idx] = gpuGlobalDataCommon[38+i,idx]
    gpuGlobalDataCommon[141+i, idx] = gpuGlobalDataCommon[38+i,idx]
    for iter in 9:28
        conn = Int(gpuGlobalDataCommon[iter, idx])
        if conn == 0.0
            break
        end
        if gpuGlobalDataCommon[137+i, idx] < gpuGlobalDataCommon[38+i,conn]
            gpuGlobalDataCommon[137+i, idx] = gpuGlobalDataCommon[38+i,conn]
        end
        if gpuGlobalDataCommon[141+i, idx] > gpuGlobalDataCommon[38+i,conn]
            gpuGlobalDataCommon[141+i, idx] = gpuGlobalDataCommon[38+i,conn]
        end
    end
    return nothing
end

# function q_var_derivatives(globaldata, configData)
#     power::Float64 = configData["core"]["power"]

#     for (idx, itm) in enumerate(globaldata)
#         rho = itm.prim[1]
#         u1 = itm.prim[2]
#         u2 = itm.prim[3]
#         pr = itm.prim[4]

#         beta::Float64 = 0.5 * (rho / pr)
#         globaldata[idx].q[1] = log(rho) + log(beta) * 2.5 - (beta * ((u1 * u1) + (u2 * u2)))
#         two_times_beta = 2.0 * beta
#         # if idx == 1
#         #     println(globaldata[idx].q[1])
#         # end
#         globaldata[idx].q[2] = (two_times_beta * u1)
#         globaldata[idx].q[3] = (two_times_beta * u2)
#         globaldata[idx].q[4] = -two_times_beta
#         # if idx == 3
#         #     println(globaldata[3])
#         # end
#     end

#     for (idx,itm) in enumerate(globaldata)
#         x_i = itm.x
#         y_i = itm.y
#         sum_delx_sqr = zero(Float64)
#         sum_dely_sqr = zero(Float64)
#         sum_delx_dely = zero(Float64)
#         sum_delx_delq = zeros(Float64, 4)
#         sum_dely_delq = zeros(Float64, 4)
#         for conn in itm.conn
#             x_k = globaldata[conn].x
#             y_k = globaldata[conn].y
#             delx = x_k - x_i
#             dely = y_k - y_i
#             dist = hypot(delx, dely)
#             weights = dist ^ power
#             sum_delx_sqr += ((delx * delx) * weights)
#             sum_dely_sqr += ((dely * dely) * weights)
#             sum_delx_dely += ((delx * dely) * weights)
#             sum_delx_delq += (weights * delx * (globaldata[conn].q - globaldata[idx].q))
#             sum_dely_delq += (weights * dely * (globaldata[conn].q - globaldata[idx].q))
#         end
#         det = (sum_delx_sqr * sum_dely_sqr) - (sum_delx_dely * sum_delx_dely)
#         one_by_det = 1.0 / det
#         sum_delx_delq1 = sum_delx_delq * sum_dely_sqr
#         sum_dely_delq1 = sum_dely_delq * sum_delx_dely
#         tempsumx = one_by_det * (sum_delx_delq1 - sum_dely_delq1)

#         sum_dely_delq2 = sum_dely_delq * sum_delx_sqr
#         sum_delx_delq2 = sum_delx_delq * sum_delx_dely
#         tempsumy = one_by_det * (sum_dely_delq2 - sum_delx_delq2)
#         globaldata[idx].dq = [tempsumx, tempsumy]

#         for i in 1:4
#             maximum(globaldata, idx, i)
#             minimum(globaldata, idx, i)
#         end

#     end

#     # println(globaldata[3])
# end
