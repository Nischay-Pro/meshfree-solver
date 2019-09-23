function interior_dGx_pos_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataFauxFixed, gpuGlobalDataRest, gpuConfigData, numPoints, shared, flux_shared, qtilde_shared)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    thread_idx = threadIdx().x
    block_dim = blockDim().x

    sum_delx_sqr = 0.0
    sum_dely_sqr = 0.0
    sum_delx_dely = 0.0
    sum_delx_delf = SVector{4,Float64}(0, 0, 0, 0)
    sum_dely_delf = SVector{4,Float64}(0, 0, 0, 0)

    # phi_i = SVector{4,Float64}(0, 0, 0, 0)
    # phi_k = SVector{4,Float64}(0, 0, 0, 0)

    x_i = gpuGlobalDataFauxFixed[idx]
    y_i = gpuGlobalDataFauxFixed[idx + numPoints]
    nx = gpuGlobalDataFauxFixed[idx + 2 * numPoints]
    ny = gpuGlobalDataFauxFixed[idx + 3 * numPoints]


    # limiter_flag = gpuConfigData[7]
    power = gpuConfigData[6]
    # gamma = gpuConfigData[15]
    for iter in 15:24
        conn = gpuGlobalDataConn[idx, iter]
        if conn == 0.0
            break
        end

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
        flux_Gxp_kernel(nx, ny, idx, shared, +)
        venkat_limiter_kernel_qtilde(gpuGlobalDataFixedPoint, gpuGlobalDataRest, conn, gpuConfigData, delx, dely, shared, qtilde_shared)
        flux_Gxp_kernel(nx, ny, idx, shared, -)

        # CUDAnative.synchronize()
        temp_var = @SVector [shared[thread_idx], shared[thread_idx + block_dim], shared[thread_idx + block_dim * 2], shared[thread_idx + block_dim * 3]]
        sum_delx_delf += temp_var * dels_weights
        sum_dely_delf += temp_var * deln_weights
    end

    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1.0 / det

    flux_shared[thread_idx] += (sum_delx_delf[1]*sum_dely_sqr - sum_dely_delf[1]*sum_delx_dely)*one_by_det
    flux_shared[thread_idx + block_dim] += (sum_delx_delf[2]*sum_dely_sqr - sum_dely_delf[2]*sum_delx_dely)*one_by_det
    flux_shared[thread_idx + block_dim * 2] += (sum_delx_delf[3]*sum_dely_sqr - sum_dely_delf[3]*sum_delx_dely)*one_by_det
    flux_shared[thread_idx + block_dim * 3] += (sum_delx_delf[4]*sum_dely_sqr - sum_dely_delf[4]*sum_delx_dely)*one_by_det

    return nothing
end

function interior_dGx_neg_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataFauxFixed, gpuGlobalDataRest, gpuConfigData, numPoints, shared, flux_shared, qtilde_shared)
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
    for iter in 25:34
        conn = gpuGlobalDataConn[idx, iter]
        if conn == 0.0
            break
        end

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
        flux_Gxn_kernel(nx, ny, idx, shared, +)
        venkat_limiter_kernel_qtilde(gpuGlobalDataFixedPoint, gpuGlobalDataRest, conn, gpuConfigData, delx, dely, shared, qtilde_shared)
        flux_Gxn_kernel(nx, ny, idx, shared, -)
        # CUDAnative.synchronize()
        temp_var = @SVector [shared[thread_idx], shared[thread_idx + block_dim], shared[thread_idx + block_dim * 2], shared[thread_idx + block_dim * 3] ]
        sum_delx_delf += temp_var * dels_weights
        sum_dely_delf += temp_var * deln_weights
    end

    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1.0 / det
    flux_shared[thread_idx] += (sum_delx_delf[1]*sum_dely_sqr - sum_dely_delf[1]*sum_delx_dely)*one_by_det
    flux_shared[thread_idx + block_dim] += (sum_delx_delf[2]*sum_dely_sqr - sum_dely_delf[2]*sum_delx_dely)*one_by_det
    flux_shared[thread_idx + block_dim * 2] += (sum_delx_delf[3]*sum_dely_sqr - sum_dely_delf[3]*sum_delx_dely)*one_by_det
    flux_shared[thread_idx + block_dim * 3] += (sum_delx_delf[4]*sum_dely_sqr - sum_dely_delf[4]*sum_delx_dely)*one_by_det

    return nothing
end

function interior_dGy_pos_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataFauxFixed, gpuGlobalDataRest, gpuConfigData, numPoints, shared, flux_shared, qtilde_shared)
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

    for iter in 35:44
        conn = gpuGlobalDataConn[idx, iter]
        if conn == 0.0
            break
        end

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
        flux_Gyp_kernel(nx, ny, idx, shared, +)
        venkat_limiter_kernel_qtilde(gpuGlobalDataFixedPoint, gpuGlobalDataRest, conn, gpuConfigData, delx, dely, shared, qtilde_shared)
        flux_Gyp_kernel(nx, ny, idx, shared, -)
        # CUDAnative.synchronize()
        temp_var = @SVector [shared[thread_idx], shared[thread_idx + block_dim], shared[thread_idx + block_dim * 2], shared[thread_idx + block_dim * 3] ]
        sum_delx_delf += temp_var * dels_weights
        sum_dely_delf += temp_var * deln_weights

    end

    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1.0 / det
    flux_shared[thread_idx] += (sum_dely_delf[1]*sum_delx_sqr - sum_delx_delf[1]*sum_delx_dely)*one_by_det
    flux_shared[thread_idx + block_dim] += (sum_dely_delf[2]*sum_delx_sqr - sum_delx_delf[2]*sum_delx_dely)*one_by_det
    flux_shared[thread_idx + block_dim * 2] += (sum_dely_delf[3]*sum_delx_sqr - sum_delx_delf[3]*sum_delx_dely)*one_by_det
    flux_shared[thread_idx + block_dim * 3] += (sum_dely_delf[4]*sum_delx_sqr - sum_delx_delf[4]*sum_delx_dely)*one_by_det

    return nothing
end

function interior_dGy_neg_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataFauxFixed, gpuGlobalDataRest, gpuConfigData, numPoints, shared, flux_shared, qtilde_shared)
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
        temp_var = @SVector [shared[thread_idx], shared[thread_idx + block_dim], shared[thread_idx + block_dim * 2], shared[thread_idx + block_dim * 3] ]
        sum_delx_delf += temp_var * dels_weights
        sum_dely_delf += temp_var * deln_weights

    end

    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1.0 / det
    flux_shared[thread_idx] += (sum_dely_delf[1]*sum_delx_sqr - sum_delx_delf[1]*sum_delx_dely)*one_by_det
    flux_shared[thread_idx + block_dim] += (sum_dely_delf[2]*sum_delx_sqr - sum_delx_delf[2]*sum_delx_dely)*one_by_det
    flux_shared[thread_idx + block_dim * 2] += (sum_dely_delf[3]*sum_delx_sqr - sum_delx_delf[3]*sum_delx_dely)*one_by_det
    flux_shared[thread_idx + block_dim * 3] += (sum_dely_delf[4]*sum_delx_sqr - sum_delx_delf[4]*sum_delx_dely)*one_by_det

    return nothing
end
