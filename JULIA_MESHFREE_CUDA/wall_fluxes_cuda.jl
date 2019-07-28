function wall_dGx_pos_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, gpuConfigData, shared, flux_shared)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    thread_idx = threadIdx().x
    block_dim = blockDim().x

    sum_delx_sqr = 0.0
    sum_dely_sqr = 0.0
    sum_delx_dely = 0.0
    sum_delx_delf = SVector{4,Float64}(0, 0, 0, 0)
    sum_dely_delf = SVector{4,Float64}(0, 0, 0, 0)
    qtilde_i = SVector{4,Float64}(0, 0, 0, 0)
    qtilde_k = SVector{4,Float64}(0, 0, 0, 0)

    x_i = gpuGlobalDataFixedPoint[idx].x
    y_i = gpuGlobalDataFixedPoint[idx].y
    nx = gpuGlobalDataFixedPoint[idx].nx
    ny = gpuGlobalDataFixedPoint[idx].ny


    # power = gpuConfigData[6]
    # gamma = gpuConfigData[15]
    for iter in 15:24
        conn = gpuGlobalDataConn[iter, idx]
        if conn == 0
            break
        end

        # x_k = gpuGlobalDataFixedPoint[conn].x
        # y_k = gpuGlobalDataFixedPoint[conn].y
        delx = gpuGlobalDataFixedPoint[conn].x - x_i
        dely = gpuGlobalDataFixedPoint[conn].y - y_i
        dels = delx*ny - dely*nx
        deln = delx*nx + dely*ny
        # dist = CUDAnative.hypot(dels, deln)
        # weights = CUDAnative.pow(dist, gpuConfigData[6])
        weights = 1.0


        dels_weights = dels*weights
        deln_weights = deln*weights

        sum_delx_sqr += dels*dels_weights
        sum_dely_sqr += deln*deln_weights
        sum_delx_dely += dels*deln_weights

        # if idx == 3
        #     @cuprintf("\n %d", conn)
        #     @cuprintf("\n %.17f %.17f %.17f %.17f", qtilde_k[1], qtilde_k[2], qtilde_k[3], qtilde_k[4])
        # end


        # if limiter_flag == 1
            venkat_limiter_kernel(gpuGlobalDataFixedPoint, gpuGlobalDataRest, idx, gpuConfigData, delx, dely, shared, thread_idx, block_dim)
            qtilde_i = @SVector [ gpuGlobalDataRest[9, idx] - 0.5*shared[thread_idx]*(delx * gpuGlobalDataRest[13, idx] + dely * gpuGlobalDataRest[17, idx]),
                gpuGlobalDataRest[10, idx] - 0.5*shared[thread_idx + block_dim]*(delx * gpuGlobalDataRest[14, idx] + dely * gpuGlobalDataRest[18, idx]),
                gpuGlobalDataRest[11, idx] - 0.5*shared[thread_idx + block_dim * 2]*(delx * gpuGlobalDataRest[15, idx] + dely * gpuGlobalDataRest[19, idx]),
                gpuGlobalDataRest[12, idx] - 0.5*shared[thread_idx + block_dim * 3]*(delx * gpuGlobalDataRest[16, idx] + dely * gpuGlobalDataRest[20, idx])]
            venkat_limiter_kernel(gpuGlobalDataFixedPoint, gpuGlobalDataRest, conn, gpuConfigData, delx, dely, shared, thread_idx, block_dim)
            qtilde_k = @SVector [ gpuGlobalDataRest[9, conn] - 0.5*shared[thread_idx]*(delx * gpuGlobalDataRest[13, conn] + dely * gpuGlobalDataRest[17, conn]),
                gpuGlobalDataRest[10, conn] - 0.5*shared[thread_idx + block_dim]*(delx * gpuGlobalDataRest[14, conn] + dely * gpuGlobalDataRest[18, conn]),
                gpuGlobalDataRest[11, conn] - 0.5*shared[thread_idx + block_dim * 2]*(delx * gpuGlobalDataRest[15, conn] + dely * gpuGlobalDataRest[19, conn]),
                gpuGlobalDataRest[12, conn] - 0.5*shared[thread_idx + block_dim * 3]*(delx * gpuGlobalDataRest[16, conn] + dely * gpuGlobalDataRest[20, conn])]
        # end


        shared[thread_idx], shared[thread_idx + block_dim], shared[thread_idx + block_dim * 2], shared[thread_idx + block_dim * 3] = 0.0, 0.0, 0.0, 0.0

        qtilde_to_primitive_kernel(qtilde_i, gpuConfigData, shared, thread_idx, block_dim)
        flux_quad_GxII_kernel(nx, ny, idx, shared, +, thread_idx, block_dim)
        qtilde_to_primitive_kernel(qtilde_k, gpuConfigData, shared, thread_idx, block_dim)
        flux_quad_GxII_kernel(nx, ny, idx, shared, -, thread_idx, block_dim)

        temp_var = @SVector [shared[thread_idx], shared[thread_idx + block_dim], shared[thread_idx + block_dim * 2], shared[thread_idx + block_dim * 3] ]
        sum_delx_delf += temp_var * dels_weights
        sum_dely_delf += temp_var * deln_weights

    end

    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1.0 / det
    # if idx ==3
    #     @cuprintf("\n===Gn===")
    #     @cuprintf("\n %.17f %.17f %.17f %.17f", sum_1,sum_2,sum_3,sum_4)
    #     @cuprintf("\n %.17f %.17f %.17f %.17f", sum_5,sum_6,sum_7,sum_8)
    # end
    flux_shared[thread_idx] += 2 * (sum_delx_delf[1]*sum_dely_sqr - sum_dely_delf[1]*sum_delx_dely)*one_by_det
    flux_shared[thread_idx + block_dim] += 2 * (sum_delx_delf[2]*sum_dely_sqr - sum_dely_delf[2]*sum_delx_dely)*one_by_det
    flux_shared[thread_idx + block_dim * 2] += 2 * (sum_delx_delf[3]*sum_dely_sqr - sum_dely_delf[3]*sum_delx_dely)*one_by_det
    flux_shared[thread_idx + block_dim * 3] += 2 * (sum_delx_delf[4]*sum_dely_sqr - sum_dely_delf[4]*sum_delx_dely)*one_by_det
    # if idx ==3
    #     @cuprintf("\n %.17f %.17f %.17f %.17f", gpuGlobalDataRest[5, idx],gpuGlobalDataRest[6, idx],gpuGlobalDataRest[7, idx],gpuGlobalDataRest[8, idx])
    # end
    return nothing
end

function wall_dGx_neg_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, gpuConfigData, shared, flux_shared)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    thread_idx = threadIdx().x
    block_dim = blockDim().x

    sum_delx_sqr = 0.0
    sum_dely_sqr = 0.0
    sum_delx_dely = 0.0
    sum_delx_delf = SVector{4,Float64}(0, 0, 0, 0)
    sum_dely_delf = SVector{4,Float64}(0, 0, 0, 0)
    qtilde_i = SVector{4,Float64}(0, 0, 0, 0)
    qtilde_k = SVector{4,Float64}(0, 0, 0, 0)

    x_i = gpuGlobalDataFixedPoint[idx].x
    y_i = gpuGlobalDataFixedPoint[idx].y
    nx = gpuGlobalDataFixedPoint[idx].nx
    ny = gpuGlobalDataFixedPoint[idx].ny


    # power = gpuConfigData[6]
    # gamma = gpuConfigData[15]
    for iter in 25:34
        conn = gpuGlobalDataConn[iter, idx]
        if conn == 0
            break
        end

        # x_k = gpuGlobalDataFixedPoint[conn].x
        # y_k = gpuGlobalDataFixedPoint[conn].y
        delx = gpuGlobalDataFixedPoint[conn].x - x_i
        dely = gpuGlobalDataFixedPoint[conn].y - y_i
        dels = delx*ny - dely*nx
        deln = delx*nx + dely*ny
        # dist = CUDAnative.hypot(dels, deln)
        # weights = CUDAnative.pow(dist, gpuConfigData[6])
        weights = 1.0


        dels_weights = dels*weights
        deln_weights = deln*weights

        sum_delx_sqr += dels*dels_weights
        sum_dely_sqr += deln*deln_weights
        sum_delx_dely += dels*deln_weights

        # if limiter_flag == 1
            venkat_limiter_kernel(gpuGlobalDataFixedPoint, gpuGlobalDataRest, idx, gpuConfigData, delx, dely, shared, thread_idx, block_dim)
            qtilde_i = @SVector [ gpuGlobalDataRest[9, idx] - 0.5*shared[thread_idx]*(delx * gpuGlobalDataRest[13, idx] + dely * gpuGlobalDataRest[17, idx]),
                gpuGlobalDataRest[10, idx] - 0.5*shared[thread_idx + block_dim]*(delx * gpuGlobalDataRest[14, idx] + dely * gpuGlobalDataRest[18, idx]),
                gpuGlobalDataRest[11, idx] - 0.5*shared[thread_idx + block_dim * 2]*(delx * gpuGlobalDataRest[15, idx] + dely * gpuGlobalDataRest[19, idx]),
                gpuGlobalDataRest[12, idx] - 0.5*shared[thread_idx + block_dim * 3]*(delx * gpuGlobalDataRest[16, idx] + dely * gpuGlobalDataRest[20, idx])]
            venkat_limiter_kernel(gpuGlobalDataFixedPoint, gpuGlobalDataRest, conn, gpuConfigData, delx, dely, shared, thread_idx, block_dim)
            qtilde_k = @SVector [ gpuGlobalDataRest[9, conn] - 0.5*shared[thread_idx]*(delx * gpuGlobalDataRest[13, conn] + dely * gpuGlobalDataRest[17, conn]),
                gpuGlobalDataRest[10, conn] - 0.5*shared[thread_idx + block_dim]*(delx * gpuGlobalDataRest[14, conn] + dely * gpuGlobalDataRest[18, conn]),
                gpuGlobalDataRest[11, conn] - 0.5*shared[thread_idx + block_dim * 2]*(delx * gpuGlobalDataRest[15, conn] + dely * gpuGlobalDataRest[19, conn]),
                gpuGlobalDataRest[12, conn] - 0.5*shared[thread_idx + block_dim * 3]*(delx * gpuGlobalDataRest[16, conn] + dely * gpuGlobalDataRest[20, conn])]
        # end


        shared[thread_idx], shared[thread_idx + block_dim], shared[thread_idx + block_dim * 2], shared[thread_idx + block_dim * 3] = 0.0, 0.0, 0.0, 0.0

        qtilde_to_primitive_kernel(qtilde_i, gpuConfigData, shared, thread_idx, block_dim)
        flux_quad_GxI_kernel(nx, ny, idx, shared, +, thread_idx, block_dim)
        qtilde_to_primitive_kernel(qtilde_k, gpuConfigData, shared, thread_idx, block_dim)
        flux_quad_GxI_kernel(nx, ny, idx, shared, -, thread_idx, block_dim)
        # CUDAnative.synchronize()
        temp_var = @SVector [shared[thread_idx], shared[thread_idx + block_dim], shared[thread_idx + block_dim * 2], shared[thread_idx + block_dim * 3]]
        sum_delx_delf += temp_var * dels_weights
        sum_dely_delf += temp_var * deln_weights
        # if idx == 3
        #     @cuprintf("\n *** ")
        #     @cuprintf("\n %f %f %f %f", gpuGlobalDataRest[37, idx], gpuGlobalDataRest[38, idx], gpuGlobalDataRest[39, idx], gpuGlobalDataRest[40, idx])
        #     @cuprintf("\n %f %f %f %f", gpuGlobalDataRest[41, idx], gpuGlobalDataRest[42, idx], gpuGlobalDataRest[43, idx], gpuGlobalDataRest[44, idx])
        #     @cuprintf("\n %f", dels_weights)
        #     @cuprintf("\n %f", deln_weights)
        # end
    end

    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1.0 / det

    flux_shared[thread_idx] += 2 * (sum_delx_delf[1]*sum_dely_sqr - sum_dely_delf[1]*sum_delx_dely)*one_by_det
    flux_shared[thread_idx + block_dim] += 2 * (sum_delx_delf[2]*sum_dely_sqr - sum_dely_delf[2]*sum_delx_dely)*one_by_det
    flux_shared[thread_idx + block_dim * 2] += 2 * (sum_delx_delf[3]*sum_dely_sqr - sum_dely_delf[3]*sum_delx_dely)*one_by_det
    flux_shared[thread_idx + block_dim * 3] += 2 * (sum_delx_delf[4]*sum_dely_sqr - sum_dely_delf[4]*sum_delx_dely)*one_by_det
    # if idx ==3
    #     @cuprintf("\n %f %f %f %f", gpuGlobalDataRest[5, idx],gpuGlobalDataRest[6, idx],gpuGlobalDataRest[7, idx],gpuGlobalDataRest[8, idx])
    # end
    return nothing
end

function wall_dGy_neg_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, gpuConfigData, shared, flux_shared)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    thread_idx = threadIdx().x
    block_dim = blockDim().x

    sum_delx_sqr = 0.0
    sum_dely_sqr = 0.0
    sum_delx_dely = 0.0
    sum_delx_delf = SVector{4,Float64}(0, 0, 0, 0)
    sum_dely_delf = SVector{4,Float64}(0, 0, 0, 0)
    qtilde_i = SVector{4,Float64}(0, 0, 0, 0)
    qtilde_k = SVector{4,Float64}(0, 0, 0, 0)


    x_i = gpuGlobalDataFixedPoint[idx].x
    y_i = gpuGlobalDataFixedPoint[idx].y
    nx = gpuGlobalDataFixedPoint[idx].nx
    ny = gpuGlobalDataFixedPoint[idx].ny



    # power = gpuConfigData[6]
    # gamma = gpuConfigData[15]

    for iter in 45:54
        conn = gpuGlobalDataConn[iter, idx]
        if conn == 0
            break
        end

        # x_k = gpuGlobalDataFixedPoint[conn].x
        # y_k = gpuGlobalDataFixedPoint[conn].y
        delx = gpuGlobalDataFixedPoint[conn].x - x_i
        dely = gpuGlobalDataFixedPoint[conn].y - y_i
        dels = delx*ny - dely*nx
        deln = delx*nx + dely*ny
        # dist = CUDAnative.hypot(dels, deln)
        # weights = CUDAnative.pow(dist, gpuConfigData[6])
        weights = 1.0


        dels_weights = dels*weights
        deln_weights = deln*weights

        sum_delx_sqr += dels*dels_weights
        sum_dely_sqr += deln*deln_weights
        sum_delx_dely += dels*deln_weights

        # if limiter_flag == 1
            venkat_limiter_kernel(gpuGlobalDataFixedPoint, gpuGlobalDataRest, idx, gpuConfigData, delx, dely, shared, thread_idx, block_dim)
            qtilde_i = @SVector [ gpuGlobalDataRest[9, idx] - 0.5*shared[thread_idx]*(delx * gpuGlobalDataRest[13, idx] + dely * gpuGlobalDataRest[17, idx]),
                gpuGlobalDataRest[10, idx] - 0.5*shared[thread_idx + block_dim]*(delx * gpuGlobalDataRest[14, idx] + dely * gpuGlobalDataRest[18, idx]),
                gpuGlobalDataRest[11, idx] - 0.5*shared[thread_idx + block_dim * 2]*(delx * gpuGlobalDataRest[15, idx] + dely * gpuGlobalDataRest[19, idx]),
                gpuGlobalDataRest[12, idx] - 0.5*shared[thread_idx + block_dim * 3]*(delx * gpuGlobalDataRest[16, idx] + dely * gpuGlobalDataRest[20, idx])]
            venkat_limiter_kernel(gpuGlobalDataFixedPoint, gpuGlobalDataRest, conn, gpuConfigData, delx, dely, shared, thread_idx, block_dim)
            qtilde_k = @SVector [ gpuGlobalDataRest[9, conn] - 0.5*shared[thread_idx]*(delx * gpuGlobalDataRest[13, conn] + dely * gpuGlobalDataRest[17, conn]),
                gpuGlobalDataRest[10, conn] - 0.5*shared[thread_idx + block_dim]*(delx * gpuGlobalDataRest[14, conn] + dely * gpuGlobalDataRest[18, conn]),
                gpuGlobalDataRest[11, conn] - 0.5*shared[thread_idx + block_dim * 2]*(delx * gpuGlobalDataRest[15, conn] + dely * gpuGlobalDataRest[19, conn]),
                gpuGlobalDataRest[12, conn] - 0.5*shared[thread_idx + block_dim * 3]*(delx * gpuGlobalDataRest[16, conn] + dely * gpuGlobalDataRest[20, conn])]
        # end


        shared[thread_idx], shared[thread_idx + block_dim], shared[thread_idx + block_dim * 2], shared[thread_idx + block_dim * 3] = 0.0, 0.0, 0.0, 0.0

        qtilde_to_primitive_kernel(qtilde_i, gpuConfigData, shared, thread_idx, block_dim)
        flux_Gyn_kernel(nx, ny, idx, shared, +, thread_idx, block_dim)
        qtilde_to_primitive_kernel(qtilde_k, gpuConfigData, shared, thread_idx, block_dim)
        flux_Gyn_kernel(nx, ny, idx, shared, -, thread_idx, block_dim)
        # CUDAnative.synchronize()
        temp_var = @SVector [shared[thread_idx], shared[thread_idx + block_dim], shared[thread_idx + block_dim * 2], shared[thread_idx + block_dim * 3]]
        sum_delx_delf += temp_var * dels_weights
        sum_dely_delf += temp_var * deln_weights

    end

    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1.0 / det

    flux_shared[thread_idx] += 2 * (sum_dely_delf[1]*sum_delx_sqr - sum_delx_delf[1]*sum_delx_dely)*one_by_det
    flux_shared[thread_idx + block_dim] += 2 * (sum_dely_delf[2]*sum_delx_sqr - sum_delx_delf[2]*sum_delx_dely)*one_by_det
    flux_shared[thread_idx + block_dim * 2] += 2 * (sum_dely_delf[3]*sum_delx_sqr - sum_delx_delf[3]*sum_delx_dely)*one_by_det
    flux_shared[thread_idx + block_dim * 3] += 2 * (sum_dely_delf[4]*sum_delx_sqr - sum_delx_delf[4]*sum_delx_dely)*one_by_det

    return nothing
end