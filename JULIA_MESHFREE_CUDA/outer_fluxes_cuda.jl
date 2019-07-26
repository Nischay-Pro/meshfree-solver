function outer_dGx_pos_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, gpuConfigData, shared, flux_shared)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    thread_idx = (Int(threadIdx().x) - 1) * 8

    sum_delx_sqr = 0.0
    sum_dely_sqr = 0.0
    sum_delx_dely = 0.0
    sum_delx_delf = SVector{4,Float64}(0, 0, 0, 0)
    sum_dely_delf = SVector{4,Float64}(0, 0, 0, 0)

    x_i = gpuGlobalDataFixedPoint[idx].x
    y_i = gpuGlobalDataFixedPoint[idx].y
    nx = gpuGlobalDataFixedPoint[idx].nx
    ny = gpuGlobalDataFixedPoint[idx].ny

    tx = ny
    ty = -nx

    power = gpuConfigData[6]
    gamma = gpuConfigData[15]

    for iter in 15:24
        conn = gpuGlobalDataConn[iter, idx]
        if conn == 0
            break
        end

        delx = gpuGlobalDataFixedPoint[conn].x - x_i
        dely = gpuGlobalDataFixedPoint[conn].y - y_i
        dels = delx*tx + dely*ty
        deln = delx*nx + dely*ny
        dist = CUDAnative.hypot(dels, deln)
        weights = CUDAnative.pow(dist, power)
        dels_weights = dels*weights
        deln_weights = deln*weights

        sum_delx_sqr += dels*dels_weights
        sum_dely_sqr += deln*deln_weights
        sum_delx_dely += dels*deln_weights

        # if limiter_flag == 1
            venkat_limiter_kernel(gpuGlobalDataFixedPoint, gpuGlobalDataRest, idx, gpuConfigData, delx, dely, shared, thread_idx)
            qtilde_i1, qtilde_i2, qtilde_i3, qtilde_i4 =
                gpuGlobalDataRest[9, idx] - 0.5*shared[thread_idx + 1]*(delx * gpuGlobalDataRest[13, idx] + dely * gpuGlobalDataRest[17, idx]),
                gpuGlobalDataRest[10, idx] - 0.5*shared[thread_idx + 2]*(delx * gpuGlobalDataRest[14, idx] + dely * gpuGlobalDataRest[18, idx]),
                gpuGlobalDataRest[11, idx] - 0.5*shared[thread_idx + 3]*(delx * gpuGlobalDataRest[15, idx] + dely * gpuGlobalDataRest[19, idx]),
                gpuGlobalDataRest[12, idx] - 0.5*shared[thread_idx + 4]*(delx * gpuGlobalDataRest[16, idx] + dely * gpuGlobalDataRest[20, idx])
            venkat_limiter_kernel(gpuGlobalDataFixedPoint, gpuGlobalDataRest, conn, gpuConfigData, delx, dely, shared, thread_idx)
            qtilde_k1, qtilde_k2, qtilde_k3, qtilde_k4 =
                gpuGlobalDataRest[9, conn] - 0.5*shared[thread_idx + 1]*(delx * gpuGlobalDataRest[13, conn] + dely * gpuGlobalDataRest[17, conn]),
                gpuGlobalDataRest[10, conn] - 0.5*shared[thread_idx + 2]*(delx * gpuGlobalDataRest[14, conn] + dely * gpuGlobalDataRest[18, conn]),
                gpuGlobalDataRest[11, conn] - 0.5*shared[thread_idx + 3]*(delx * gpuGlobalDataRest[15, conn] + dely * gpuGlobalDataRest[19, conn]),
                gpuGlobalDataRest[12, conn] - 0.5*shared[thread_idx + 4]*(delx * gpuGlobalDataRest[16, conn] + dely * gpuGlobalDataRest[20, conn])
        # end


        for shared_iter in 1:4
            shared[thread_idx + shared_iter] = 0.0
        end

        qtilde_to_primitive_kernel(qtilde_i1, qtilde_i2, qtilde_i3, qtilde_i4, gamma, shared, thread_idx)
        flux_quad_GxIII_kernel(nx, ny, idx, shared, +, thread_idx)
        qtilde_to_primitive_kernel(qtilde_k1, qtilde_k2, qtilde_k3, qtilde_k4, gamma, shared, thread_idx)
        flux_quad_GxIII_kernel(nx, ny, idx, shared, -, thread_idx)
        # CUDAnative.synchronize()
        temp_var = @SVector [shared[thread_idx + 1], shared[thread_idx + 2], shared[thread_idx + 3], shared[thread_idx + 4] ]
        sum_delx_delf += temp_var * dels_weights
        sum_dely_delf += temp_var * deln_weights
    end

    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1.0 / det

    flux_shared[thread_idx + 1] += (sum_delx_delf[1]*sum_dely_sqr - sum_dely_delf[1]*sum_delx_dely)*one_by_det
    flux_shared[thread_idx + 2] += (sum_delx_delf[2]*sum_dely_sqr - sum_dely_delf[2]*sum_delx_dely)*one_by_det
    flux_shared[thread_idx + 3] += (sum_delx_delf[3]*sum_dely_sqr - sum_dely_delf[3]*sum_delx_dely)*one_by_det
    flux_shared[thread_idx + 4] += (sum_delx_delf[4]*sum_dely_sqr - sum_dely_delf[4]*sum_delx_dely)*one_by_det
    return nothing
end

function outer_dGx_neg_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, gpuConfigData, shared, flux_shared)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    thread_idx = (Int(threadIdx().x) - 1) * 8

    sum_delx_sqr = 0.0
    sum_dely_sqr = 0.0
    sum_delx_dely = 0.0
    sum_delx_delf = SVector{4,Float64}(0, 0, 0, 0)
    sum_dely_delf = SVector{4,Float64}(0, 0, 0, 0)

    x_i = gpuGlobalDataFixedPoint[idx].x
    y_i = gpuGlobalDataFixedPoint[idx].y
    nx = gpuGlobalDataFixedPoint[idx].nx
    ny = gpuGlobalDataFixedPoint[idx].ny

    tx = ny
    ty = -nx

    power = gpuConfigData[6]
    gamma = gpuConfigData[15]

    for iter in 25:34
        conn = gpuGlobalDataConn[iter, idx]
        if conn == 0
            break
        end

        # x_k = gpuGlobalDataFixedPoint[conn].x
        # y_k = gpuGlobalDataFixedPoint[conn].y
        delx = gpuGlobalDataFixedPoint[conn].x - x_i
        dely = gpuGlobalDataFixedPoint[conn].y - y_i
        dels = delx*tx + dely*ty
        deln = delx*nx + dely*ny
        dist = CUDAnative.hypot(dels, deln)
        weights = CUDAnative.pow(dist, power)
        dels_weights = dels*weights
        deln_weights = deln*weights

        sum_delx_sqr += dels*dels_weights
        sum_dely_sqr += deln*deln_weights
        sum_delx_dely += dels*deln_weights

        # if limiter_flag == 1
            venkat_limiter_kernel(gpuGlobalDataFixedPoint, gpuGlobalDataRest, idx, gpuConfigData, delx, dely, shared, thread_idx)
            qtilde_i1, qtilde_i2, qtilde_i3, qtilde_i4 =
                gpuGlobalDataRest[9, idx] - 0.5*shared[thread_idx + 1]*(delx * gpuGlobalDataRest[13, idx] + dely * gpuGlobalDataRest[17, idx]),
                gpuGlobalDataRest[10, idx] - 0.5*shared[thread_idx + 2]*(delx * gpuGlobalDataRest[14, idx] + dely * gpuGlobalDataRest[18, idx]),
                gpuGlobalDataRest[11, idx] - 0.5*shared[thread_idx + 3]*(delx * gpuGlobalDataRest[15, idx] + dely * gpuGlobalDataRest[19, idx]),
                gpuGlobalDataRest[12, idx] - 0.5*shared[thread_idx + 4]*(delx * gpuGlobalDataRest[16, idx] + dely * gpuGlobalDataRest[20, idx])
            venkat_limiter_kernel(gpuGlobalDataFixedPoint, gpuGlobalDataRest, conn, gpuConfigData, delx, dely, shared, thread_idx)
            qtilde_k1, qtilde_k2, qtilde_k3, qtilde_k4 =
                gpuGlobalDataRest[9, conn] - 0.5*shared[thread_idx + 1]*(delx * gpuGlobalDataRest[13, conn] + dely * gpuGlobalDataRest[17, conn]),
                gpuGlobalDataRest[10, conn] - 0.5*shared[thread_idx + 2]*(delx * gpuGlobalDataRest[14, conn] + dely * gpuGlobalDataRest[18, conn]),
                gpuGlobalDataRest[11, conn] - 0.5*shared[thread_idx + 3]*(delx * gpuGlobalDataRest[15, conn] + dely * gpuGlobalDataRest[19, conn]),
                gpuGlobalDataRest[12, conn] - 0.5*shared[thread_idx + 4]*(delx * gpuGlobalDataRest[16, conn] + dely * gpuGlobalDataRest[20, conn])
        # end


        for shared_iter in 1:4
            shared[thread_idx + shared_iter] = 0.0
        end

        qtilde_to_primitive_kernel(qtilde_i1, qtilde_i2, qtilde_i3, qtilde_i4, gamma, shared, thread_idx)
        flux_quad_GxIV_kernel(nx, ny, idx, shared, +, thread_idx)
        qtilde_to_primitive_kernel(qtilde_k1, qtilde_k2, qtilde_k3, qtilde_k4, gamma, shared, thread_idx)
        flux_quad_GxIV_kernel(nx, ny, idx, shared, -, thread_idx)
        # CUDAnative.synchronize()
        temp_var = @SVector [shared[thread_idx + 1], shared[thread_idx + 2], shared[thread_idx + 3], shared[thread_idx + 4] ]
        sum_delx_delf += temp_var * dels_weights
        sum_dely_delf += temp_var * deln_weights
    end

    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1.0 / det

    flux_shared[thread_idx + 1] += (sum_delx_delf[1]*sum_dely_sqr - sum_dely_delf[1]*sum_delx_dely)*one_by_det
    flux_shared[thread_idx + 2] += (sum_delx_delf[2]*sum_dely_sqr - sum_dely_delf[2]*sum_delx_dely)*one_by_det
    flux_shared[thread_idx + 3] += (sum_delx_delf[3]*sum_dely_sqr - sum_dely_delf[3]*sum_delx_dely)*one_by_det
    flux_shared[thread_idx + 4] += (sum_delx_delf[4]*sum_dely_sqr - sum_dely_delf[4]*sum_delx_dely)*one_by_det
    return nothing
end

function outer_dGy_pos_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, gpuConfigData, shared, flux_shared)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    thread_idx = (Int(threadIdx().x) - 1) * 8

    sum_delx_sqr = 0.0
    sum_dely_sqr = 0.0
    sum_delx_dely = 0.0
    sum_delx_delf = SVector{4,Float64}(0, 0, 0, 0)
    sum_dely_delf = SVector{4,Float64}(0, 0, 0, 0)

    x_i = gpuGlobalDataFixedPoint[idx].x
    y_i = gpuGlobalDataFixedPoint[idx].y
    nx = gpuGlobalDataFixedPoint[idx].nx
    ny = gpuGlobalDataFixedPoint[idx].ny

    tx = ny
    ty = -nx
    power = gpuConfigData[6]
    gamma = gpuConfigData[15]

    for iter in 35:44
        conn = gpuGlobalDataConn[iter, idx]
        if conn == 0
            break
        end

        # x_k = gpuGlobalDataFixedPoint[conn].x
        # y_k = gpuGlobalDataFixedPoint[conn].y
        delx = gpuGlobalDataFixedPoint[conn].x - x_i
        dely = gpuGlobalDataFixedPoint[conn].y - y_i
        dels = delx*tx + dely*ty
        deln = delx*nx + dely*ny
        dist = CUDAnative.hypot(dels, deln)
        weights = CUDAnative.pow(dist, power)
        dels_weights = dels*weights
        deln_weights = deln*weights

        sum_delx_sqr += dels*dels_weights
        sum_dely_sqr += deln*deln_weights
        sum_delx_dely += dels*deln_weights

        # if limiter_flag == 1
            venkat_limiter_kernel(gpuGlobalDataFixedPoint, gpuGlobalDataRest, idx, gpuConfigData, delx, dely, shared, thread_idx)
            qtilde_i1, qtilde_i2, qtilde_i3, qtilde_i4 =
                gpuGlobalDataRest[9, idx] - 0.5*shared[thread_idx + 1]*(delx * gpuGlobalDataRest[13, idx] + dely * gpuGlobalDataRest[17, idx]),
                gpuGlobalDataRest[10, idx] - 0.5*shared[thread_idx + 2]*(delx * gpuGlobalDataRest[14, idx] + dely * gpuGlobalDataRest[18, idx]),
                gpuGlobalDataRest[11, idx] - 0.5*shared[thread_idx + 3]*(delx * gpuGlobalDataRest[15, idx] + dely * gpuGlobalDataRest[19, idx]),
                gpuGlobalDataRest[12, idx] - 0.5*shared[thread_idx + 4]*(delx * gpuGlobalDataRest[16, idx] + dely * gpuGlobalDataRest[20, idx])
            venkat_limiter_kernel(gpuGlobalDataFixedPoint, gpuGlobalDataRest, conn, gpuConfigData, delx, dely, shared, thread_idx)
            qtilde_k1, qtilde_k2, qtilde_k3, qtilde_k4 =
                gpuGlobalDataRest[9, conn] - 0.5*shared[thread_idx + 1]*(delx * gpuGlobalDataRest[13, conn] + dely * gpuGlobalDataRest[17, conn]),
                gpuGlobalDataRest[10, conn] - 0.5*shared[thread_idx + 2]*(delx * gpuGlobalDataRest[14, conn] + dely * gpuGlobalDataRest[18, conn]),
                gpuGlobalDataRest[11, conn] - 0.5*shared[thread_idx + 3]*(delx * gpuGlobalDataRest[15, conn] + dely * gpuGlobalDataRest[19, conn]),
                gpuGlobalDataRest[12, conn] - 0.5*shared[thread_idx + 4]*(delx * gpuGlobalDataRest[16, conn] + dely * gpuGlobalDataRest[20, conn])
        # end


        for shared_iter in 1:4
            shared[thread_idx + shared_iter] = 0.0
        end

        qtilde_to_primitive_kernel(qtilde_i1, qtilde_i2, qtilde_i3, qtilde_i4, gamma, shared, thread_idx)
        flux_Gyp_kernel(nx, ny, idx, shared, +, thread_idx)
        qtilde_to_primitive_kernel(qtilde_k1, qtilde_k2, qtilde_k3, qtilde_k4, gamma, shared, thread_idx)
        flux_Gyp_kernel(nx, ny, idx, shared, -, thread_idx)
        # CUDAnative.synchronize()
        temp_var = @SVector [shared[thread_idx + 1], shared[thread_idx + 2], shared[thread_idx + 3], shared[thread_idx + 4] ]
        sum_delx_delf += temp_var * dels_weights
        sum_dely_delf += temp_var * deln_weights
    end

    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1.0 / det

    flux_shared[thread_idx + 1] += (sum_dely_delf[1]*sum_delx_sqr - sum_delx_delf[1]*sum_delx_dely)*one_by_det
    flux_shared[thread_idx + 2] += (sum_dely_delf[2]*sum_delx_sqr - sum_delx_delf[2]*sum_delx_dely)*one_by_det
    flux_shared[thread_idx + 3] += (sum_dely_delf[3]*sum_delx_sqr - sum_delx_delf[3]*sum_delx_dely)*one_by_det
    flux_shared[thread_idx + 4] += (sum_dely_delf[4]*sum_delx_sqr - sum_delx_delf[4]*sum_delx_dely)*one_by_det
    return nothing
end