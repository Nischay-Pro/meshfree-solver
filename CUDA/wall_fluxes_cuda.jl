function wall_dGx_pos_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataFauxFixed, gpuGlobalDataRest, gpuConfigData, numPoints, shared, flux_shared, qtilde_shared)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    thread_idx = threadIdx().x
    block_dim = blockDim().x

    sum_delx_sqr = 0.0
    sum_dely_sqr = 0.0
    sum_delx_dely = 0.0
    sum_delx_delf = SVector{4,Float64}(0, 0, 0, 0)
    sum_dely_delf = SVector{4,Float64}(0, 0, 0, 0)


    x_i = gpuGlobalDataFauxFixed[idx]
    y_i = gpuGlobalDataFauxFixed[idx + numPoints]
    nx = gpuGlobalDataFauxFixed[idx + 2 * numPoints]
    ny = gpuGlobalDataFauxFixed[idx + 3 * numPoints]


    power = gpuConfigData[6]
    # gamma = gpuConfigData[15]
    for iter in 15:24
        conn = gpuGlobalDataConn[idx, iter]
        if conn == 0.0
            break
        end

        # x_k = gpuGlobalDataFixedPoint[conn].x
        # y_k = gpuGlobalDataFixedPoint[conn].y
        delx = gpuGlobalDataFauxFixed[conn] - x_i
        dely = gpuGlobalDataFauxFixed[conn + numPoints] - y_i
        dels = delx*ny - dely*nx
        deln = delx*nx + dely*ny
        dist = CUDAnative.hypot(dels, deln)
        weights = CUDAnative.pow(dist, power)
        # weights = 1.0


        dels_weights = dels*weights
        deln_weights = deln*weights

        sum_delx_sqr += dels*dels_weights
        sum_dely_sqr += deln*deln_weights
        sum_delx_dely += dels*deln_weights

        # if idx == 3
        #     @cuprintf("\n %d", conn)
        #     @cuprintf("\n %.17f %.17f %.17f %.17f", qtilde_k[1], qtilde_k[2], qtilde_k[3], qtilde_k[4])
        # end

        shared[thread_idx], shared[thread_idx + block_dim], shared[thread_idx + block_dim * 2], shared[thread_idx + block_dim * 3] = 0.0, 0.0, 0.0, 0.0
        venkat_limiter_kernel_qtilde(gpuGlobalDataFixedPoint, gpuGlobalDataRest, idx, gpuConfigData, delx, dely, shared, qtilde_shared)
        flux_quad_GxII_kernel(nx, ny, idx, shared, +)
        venkat_limiter_kernel_qtilde(gpuGlobalDataFixedPoint, gpuGlobalDataRest, conn, gpuConfigData, delx, dely, shared, qtilde_shared)
        flux_quad_GxII_kernel(nx, ny, idx, shared, -)

        temp_var = @SVector [shared[thread_idx], shared[thread_idx + block_dim], shared[thread_idx + block_dim * 2], shared[thread_idx + block_dim * 3]]
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

function wall_dGx_neg_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataFauxFixed, gpuGlobalDataRest, gpuConfigData, numPoints, shared, flux_shared, qtilde_shared)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    thread_idx = threadIdx().x
    block_dim = blockDim().x

    sum_delx_sqr = 0.0
    sum_dely_sqr = 0.0
    sum_delx_dely = 0.0
    sum_delx_delf = SVector{4,Float64}(0, 0, 0, 0)
    sum_dely_delf = SVector{4,Float64}(0, 0, 0, 0)
    # shared[thread_idx + block_dim * 8], shared[thread_idx + block_dim * 9], shared[thread_idx + block_dim * 10], shared[thread_idx + block_dim * 11] = 0.0, 0.0, 0.0, 0.0
    # shared[thread_idx + block_dim * 12], shared[thread_idx + block_dim * 13], shared[thread_idx + block_dim * 14], shared[thread_idx + block_dim * 15] = 0.0, 0.0, 0.0, 0.0
    x_i = gpuGlobalDataFauxFixed[idx]
    y_i = gpuGlobalDataFauxFixed[idx + numPoints]
    nx = gpuGlobalDataFauxFixed[idx + 2 * numPoints]
    ny = gpuGlobalDataFauxFixed[idx + 3 * numPoints]


    power = gpuConfigData[6]
    # gamma = gpuConfigData[15]
    for iter in 25:34
        conn = gpuGlobalDataConn[idx, iter]
        if conn == 0.0
            break
        end

        # x_k = gpuGlobalDataFixedPoint[conn].x
        # y_k = gpuGlobalDataFixedPoint[conn].y
        delx = gpuGlobalDataFauxFixed[conn] - x_i
        dely = gpuGlobalDataFauxFixed[conn + numPoints] - y_i
        dels = delx*ny - dely*nx
        deln = delx*nx + dely*ny
        dist = CUDAnative.hypot(dels, deln)
        weights = CUDAnative.pow(dist, power)
        # weights = 1.0


        dels_weights = dels*weights
        deln_weights = deln*weights

        sum_delx_sqr += dels*dels_weights
        sum_dely_sqr += deln*deln_weights
        sum_delx_dely += dels*deln_weights

        shared[thread_idx], shared[thread_idx + block_dim], shared[thread_idx + block_dim * 2], shared[thread_idx + block_dim * 3] = 0.0, 0.0, 0.0, 0.0
        venkat_limiter_kernel_qtilde(gpuGlobalDataFixedPoint, gpuGlobalDataRest, idx, gpuConfigData, delx, dely, shared, qtilde_shared)
        flux_quad_GxI_kernel(nx, ny, idx, shared, +)
        venkat_limiter_kernel_qtilde(gpuGlobalDataFixedPoint, gpuGlobalDataRest, conn, gpuConfigData, delx, dely, shared, qtilde_shared)
        flux_quad_GxI_kernel(nx, ny, idx, shared, -)
        # CUDAnative.synchronize()
        temp_var = @SVector [shared[thread_idx], shared[thread_idx + block_dim], shared[thread_idx + block_dim * 2], shared[thread_idx + block_dim * 3]]
        sum_delx_delf += temp_var * dels_weights
        sum_dely_delf += temp_var * deln_weights

        # shared[thread_idx + block_dim * 8], shared[thread_idx + block_dim * 9], shared[thread_idx + block_dim * 10], shared[thread_idx + block_dim * 11] = shared[thread_idx]* dels_weights,
        # shared[thread_idx + block_dim]* dels_weights, shared[thread_idx + block_dim * 2]* dels_weights, shared[thread_idx + block_dim * 3]* dels_weights

        # shared[thread_idx + block_dim * 12], shared[thread_idx + block_dim * 13], shared[thread_idx + block_dim * 14], shared[thread_idx + block_dim * 15] = shared[thread_idx]* deln_weights,
        # shared[thread_idx + block_dim]* deln_weights, shared[thread_idx + block_dim * 2]* deln_weights, shared[thread_idx + block_dim * 3]* deln_weights

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

    # flux_shared[thread_idx] += 2 * (shared[thread_idx + block_dim * 8]*sum_dely_sqr - shared[thread_idx + block_dim * 12]*sum_delx_dely)*one_by_det
    # flux_shared[thread_idx + block_dim] += 2 * (shared[thread_idx + block_dim * 9]*sum_dely_sqr - shared[thread_idx + block_dim * 13]*sum_delx_dely)*one_by_det
    # flux_shared[thread_idx + block_dim * 2] += 2 * (shared[thread_idx + block_dim * 10]*sum_dely_sqr - shared[thread_idx + block_dim * 14]*sum_delx_dely)*one_by_det
    # flux_shared[thread_idx + block_dim * 3] += 2 * (shared[thread_idx + block_dim * 11]*sum_dely_sqr - shared[thread_idx + block_dim * 15]*sum_delx_dely)*one_by_det
    # if idx ==3
    #     @cuprintf("\n %f %f %f %f", gpuGlobalDataRest[5, idx],gpuGlobalDataRest[6, idx],gpuGlobalDataRest[7, idx],gpuGlobalDataRest[8, idx])
    # end
    return nothing
end

function wall_dGy_neg_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataFauxFixed, gpuGlobalDataRest, gpuConfigData, numPoints, shared, flux_shared, qtilde_shared)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    thread_idx = threadIdx().x
    block_dim = blockDim().x

    sum_delx_sqr = 0.0
    sum_dely_sqr = 0.0
    sum_delx_dely = 0.0
    sum_delx_delf = SVector{4,Float64}(0, 0, 0, 0)
    sum_dely_delf = SVector{4,Float64}(0, 0, 0, 0)

    x_i = gpuGlobalDataFauxFixed[idx]
    y_i = gpuGlobalDataFauxFixed[idx + numPoints]
    nx = gpuGlobalDataFauxFixed[idx + 2 * numPoints]
    ny = gpuGlobalDataFauxFixed[idx + 3 * numPoints]

    power = gpuConfigData[6]
    # gamma = gpuConfigData[15]

    for iter in 45:54
        conn = gpuGlobalDataConn[idx, iter]
        if conn == 0.0
            break
        end

        # x_k = gpuGlobalDataFixedPoint[conn].x
        # y_k = gpuGlobalDataFixedPoint[conn].y
        delx = gpuGlobalDataFauxFixed[conn] - x_i
        dely = gpuGlobalDataFauxFixed[conn + numPoints] - y_i
        dels = delx*ny - dely*nx
        deln = delx*nx + dely*ny
        dist = CUDAnative.hypot(dels, deln)
        weights = CUDAnative.pow(dist, power)
        # weights = 1.0


        dels_weights = dels*weights
        deln_weights = deln*weights

        sum_delx_sqr += dels*dels_weights
        sum_dely_sqr += deln*deln_weights
        sum_delx_dely += dels*deln_weights

        shared[thread_idx], shared[thread_idx + block_dim], shared[thread_idx + block_dim * 2], shared[thread_idx + block_dim * 3] = 0.0, 0.0, 0.0, 0.0
        venkat_limiter_kernel_qtilde(gpuGlobalDataFixedPoint, gpuGlobalDataRest, idx, gpuConfigData, delx, dely, shared, qtilde_shared)
        flux_Gyn_kernel(nx, ny, idx, shared, +)
        venkat_limiter_kernel_qtilde(gpuGlobalDataFixedPoint, gpuGlobalDataRest, conn, gpuConfigData, delx, dely, shared, qtilde_shared)
        flux_Gyn_kernel(nx, ny, idx, shared, -)
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