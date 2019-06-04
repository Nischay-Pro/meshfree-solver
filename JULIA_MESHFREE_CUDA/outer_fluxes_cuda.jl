function outer_dGx_pos_kernel(gpuGlobalDataCommon, gpuGlobalDataFixedPoint, gpuGlobalDataRest, idx, gpuConfigData)

    power = gpuConfigData[6]
    limiter_flag = gpuConfigData[7]

    sum_delx_sqr = 0.0
    sum_dely_sqr = 0.0
    sum_delx_dely = 0.0
    x_k = 0.0
    y_k = 0.0
    delx = 0.0
    dely = 0.0
    dels = 0.0
    deln = 0.0
    dist = 0.0
    weights = 0.0
    dels_weights = 0.0
    deln_weights = 0.0
    sum_delx_sqr = 0.0
    sum_dely_sqr = 0.0
    sum_delx_dely = 0.0

    for i in 146:173
        gpuGlobalDataCommon[i, idx] = 0.0
    end

    x_i = gpuGlobalDataFixedPoint[idx].x
    y_i = gpuGlobalDataFixedPoint[idx].y
    nx = gpuGlobalDataFixedPoint[idx].nx
    ny = gpuGlobalDataFixedPoint[idx].ny

    tx = ny
    ty = -nx

    for iter in 56:75
        conn = Int(gpuGlobalDataCommon[iter, idx])
        if conn == 0.0
            break
        end
        x_k = gpuGlobalDataFixedPoint[conn].x
        y_k = gpuGlobalDataFixedPoint[conn].y
        delx = x_k - x_i
        dely = y_k - y_i
        dels = delx*tx + dely*ty
        deln = delx*nx + dely*ny
        dist = CUDAnative.hypot(dels, deln)
        weights = CUDAnative.pow(dist, power)
        dels_weights = dels*weights
        deln_weights = deln*weights

        sum_delx_sqr = sum_delx_sqr + dels*dels_weights
        sum_dely_sqr = sum_dely_sqr + deln*deln_weights
        sum_delx_dely = sum_delx_dely + dels*deln_weights

        if limiter_flag == 1
            venkat_limiter_kernel_i(gpuGlobalDataCommon, gpuGlobalDataRest, idx, gpuConfigData, delx, dely)
            venkat_limiter_kernel_k(gpuGlobalDataCommon, gpuGlobalDataRest, conn, gpuConfigData, idx, delx, dely)
            # CUDAnative.synchronize()
            qtilde_i =  (
                            gpuGlobalDataRest[9, idx] - 0.5*gpuGlobalDataRest[29, idx]*(delx * gpuGlobalDataRest[13, idx] + dely * gpuGlobalDataRest[17, idx]),
                            gpuGlobalDataRest[10, idx] - 0.5*gpuGlobalDataRest[30, idx]*(delx * gpuGlobalDataRest[14, idx] + dely * gpuGlobalDataRest[18, idx]),
                            gpuGlobalDataRest[11, idx] - 0.5*gpuGlobalDataRest[31, idx]*(delx * gpuGlobalDataRest[15, idx] + dely * gpuGlobalDataRest[19, idx]),
                            gpuGlobalDataRest[12, idx] - 0.5*gpuGlobalDataRest[32, idx]*(delx * gpuGlobalDataRest[16, idx] + dely * gpuGlobalDataRest[20, idx])
                        )
            qtilde_k = (
                            gpuGlobalDataRest[9, conn] - 0.5*gpuGlobalDataRest[33, idx]*(delx * gpuGlobalDataRest[13, conn] + dely * gpuGlobalDataRest[17, conn]),
                            gpuGlobalDataRest[10, conn] - 0.5*gpuGlobalDataRest[34, idx]*(delx * gpuGlobalDataRest[14, conn] + dely * gpuGlobalDataRest[18, conn]),
                            gpuGlobalDataRest[11, conn] - 0.5*gpuGlobalDataRest[35, idx]*(delx * gpuGlobalDataRest[15, conn] + dely * gpuGlobalDataRest[19, conn]),
                            gpuGlobalDataRest[12, conn] - 0.5*gpuGlobalDataRest[36, idx]*(delx * gpuGlobalDataRest[16, conn] + dely * gpuGlobalDataRest[20, conn])
                        )
        end

        if limiter_flag == 2
            @cuprintf("\n Havent written the code - die \n")
        end
        qtilde_to_primitive_kernel(qtilde_i, gpuConfigData, gpuGlobalDataCommon, gpuGlobalDataRest, idx)
        flux_quad_GxIII_kernel(nx, ny, gpuGlobalDataCommon, gpuGlobalDataRest, idx, 1)
        qtilde_to_primitive_kernel(qtilde_k, gpuConfigData, gpuGlobalDataCommon, gpuGlobalDataRest, idx)
        flux_quad_GxIII_kernel(nx, ny, gpuGlobalDataCommon, gpuGlobalDataRest, idx, 2)
        # CUDAnative.synchronize()
        gpuGlobalDataCommon[162, idx] += (gpuGlobalDataCommon[158, idx] - gpuGlobalDataCommon[154, idx]) * dels_weights
        gpuGlobalDataCommon[166, idx] += (gpuGlobalDataCommon[158, idx] - gpuGlobalDataCommon[154, idx]) * deln_weights
        gpuGlobalDataCommon[163, idx] += (gpuGlobalDataCommon[159, idx] - gpuGlobalDataCommon[155, idx]) * dels_weights
        gpuGlobalDataCommon[167, idx] += (gpuGlobalDataCommon[159, idx] - gpuGlobalDataCommon[155, idx]) * deln_weights
        gpuGlobalDataCommon[164, idx] += (gpuGlobalDataCommon[160, idx] - gpuGlobalDataCommon[156, idx]) * dels_weights
        gpuGlobalDataCommon[168, idx] += (gpuGlobalDataCommon[160, idx] - gpuGlobalDataCommon[156, idx]) * deln_weights
        gpuGlobalDataCommon[165, idx] += (gpuGlobalDataCommon[161, idx] - gpuGlobalDataCommon[157, idx]) * dels_weights
        gpuGlobalDataCommon[169, idx] += (gpuGlobalDataCommon[161, idx] - gpuGlobalDataCommon[157, idx]) * deln_weights
    end

    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1.0 / det

    gpuGlobalDataRest[5, idx] += (gpuGlobalDataCommon[162, idx]*sum_dely_sqr - gpuGlobalDataCommon[166, idx]*sum_delx_dely)*one_by_det
    gpuGlobalDataRest[6, idx] += (gpuGlobalDataCommon[163, idx]*sum_dely_sqr - gpuGlobalDataCommon[167, idx]*sum_delx_dely)*one_by_det
    gpuGlobalDataRest[7, idx] += (gpuGlobalDataCommon[164, idx]*sum_dely_sqr - gpuGlobalDataCommon[168, idx]*sum_delx_dely)*one_by_det
    gpuGlobalDataRest[8, idx] += (gpuGlobalDataCommon[165, idx]*sum_dely_sqr - gpuGlobalDataCommon[169, idx]*sum_delx_dely)*one_by_det
    return nothing
end

function outer_dGx_neg_kernel(gpuGlobalDataCommon, gpuGlobalDataFixedPoint, gpuGlobalDataRest, idx, gpuConfigData)

    power = gpuConfigData[6]
    limiter_flag = gpuConfigData[7]

    sum_delx_sqr = 0.0
    sum_dely_sqr = 0.0
    sum_delx_dely = 0.0
    x_k = 0.0
    y_k = 0.0
    delx = 0.0
    dely = 0.0
    dels = 0.0
    deln = 0.0
    dist = 0.0
    weights = 0.0
    dels_weights = 0.0
    deln_weights = 0.0
    sum_delx_sqr = 0.0
    sum_dely_sqr = 0.0
    sum_delx_dely = 0.0

    for i in 146:173
        gpuGlobalDataCommon[i, idx] = 0.0
    end

    x_i = gpuGlobalDataFixedPoint[idx].x
    y_i = gpuGlobalDataFixedPoint[idx].y
    nx = gpuGlobalDataFixedPoint[idx].nx
    ny = gpuGlobalDataFixedPoint[idx].ny

    tx = ny
    ty = -nx

    for iter in 76:95
        conn = Int(gpuGlobalDataCommon[iter, idx])
        if conn == 0.0
            break
        end
        x_k = gpuGlobalDataFixedPoint[conn].x
        y_k = gpuGlobalDataFixedPoint[conn].y
        delx = x_k - x_i
        dely = y_k - y_i
        dels = delx*tx + dely*ty
        deln = delx*nx + dely*ny
        dist = CUDAnative.hypot(dels, deln)
        weights = CUDAnative.pow(dist, power)
        dels_weights = dels*weights
        deln_weights = deln*weights

        sum_delx_sqr = sum_delx_sqr + dels*dels_weights
        sum_dely_sqr = sum_dely_sqr + deln*deln_weights
        sum_delx_dely = sum_delx_dely + dels*deln_weights

        if limiter_flag == 1
            venkat_limiter_kernel_i(gpuGlobalDataCommon, gpuGlobalDataRest, idx, gpuConfigData, delx, dely)
            venkat_limiter_kernel_k(gpuGlobalDataCommon, gpuGlobalDataRest, conn, gpuConfigData, idx, delx, dely)
            # CUDAnative.synchronize()
            qtilde_i =  (
                            gpuGlobalDataRest[9, idx] - 0.5*gpuGlobalDataRest[29, idx]*(delx * gpuGlobalDataRest[13, idx] + dely * gpuGlobalDataRest[17, idx]),
                            gpuGlobalDataRest[10, idx] - 0.5*gpuGlobalDataRest[30, idx]*(delx * gpuGlobalDataRest[14, idx] + dely * gpuGlobalDataRest[18, idx]),
                            gpuGlobalDataRest[11, idx] - 0.5*gpuGlobalDataRest[31, idx]*(delx * gpuGlobalDataRest[15, idx] + dely * gpuGlobalDataRest[19, idx]),
                            gpuGlobalDataRest[12, idx] - 0.5*gpuGlobalDataRest[32, idx]*(delx * gpuGlobalDataRest[16, idx] + dely * gpuGlobalDataRest[20, idx])
                        )
            qtilde_k = (
                            gpuGlobalDataRest[9, conn] - 0.5*gpuGlobalDataRest[33, idx]*(delx * gpuGlobalDataRest[13, conn] + dely * gpuGlobalDataRest[17, conn]),
                            gpuGlobalDataRest[10, conn] - 0.5*gpuGlobalDataRest[34, idx]*(delx * gpuGlobalDataRest[14, conn] + dely * gpuGlobalDataRest[18, conn]),
                            gpuGlobalDataRest[11, conn] - 0.5*gpuGlobalDataRest[35, idx]*(delx * gpuGlobalDataRest[15, conn] + dely * gpuGlobalDataRest[19, conn]),
                            gpuGlobalDataRest[12, conn] - 0.5*gpuGlobalDataRest[36, idx]*(delx * gpuGlobalDataRest[16, conn] + dely * gpuGlobalDataRest[20, conn])
                        )
        end

        if limiter_flag == 2
            @cuprintf("\n Havent written the code - die \n")
        end
        qtilde_to_primitive_kernel(qtilde_i, gpuConfigData, gpuGlobalDataCommon, gpuGlobalDataRest, idx)
        flux_quad_GxIV_kernel(nx, ny, gpuGlobalDataCommon, gpuGlobalDataRest, idx, 1)
        qtilde_to_primitive_kernel(qtilde_k, gpuConfigData, gpuGlobalDataCommon, gpuGlobalDataRest, idx)
        flux_quad_GxIV_kernel(nx, ny, gpuGlobalDataCommon, gpuGlobalDataRest, idx, 2)
        # CUDAnative.synchronize()
        gpuGlobalDataCommon[162, idx] += (gpuGlobalDataCommon[158, idx] - gpuGlobalDataCommon[154, idx]) * dels_weights
        gpuGlobalDataCommon[166, idx] += (gpuGlobalDataCommon[158, idx] - gpuGlobalDataCommon[154, idx]) * deln_weights
        gpuGlobalDataCommon[163, idx] += (gpuGlobalDataCommon[159, idx] - gpuGlobalDataCommon[155, idx]) * dels_weights
        gpuGlobalDataCommon[167, idx] += (gpuGlobalDataCommon[159, idx] - gpuGlobalDataCommon[155, idx]) * deln_weights
        gpuGlobalDataCommon[164, idx] += (gpuGlobalDataCommon[160, idx] - gpuGlobalDataCommon[156, idx]) * dels_weights
        gpuGlobalDataCommon[168, idx] += (gpuGlobalDataCommon[160, idx] - gpuGlobalDataCommon[156, idx]) * deln_weights
        gpuGlobalDataCommon[165, idx] += (gpuGlobalDataCommon[161, idx] - gpuGlobalDataCommon[157, idx]) * dels_weights
        gpuGlobalDataCommon[169, idx] += (gpuGlobalDataCommon[161, idx] - gpuGlobalDataCommon[157, idx]) * deln_weights
    end

    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1.0 / det

    gpuGlobalDataRest[5, idx] += (gpuGlobalDataCommon[162, idx]*sum_dely_sqr - gpuGlobalDataCommon[166, idx]*sum_delx_dely)*one_by_det
    gpuGlobalDataRest[6, idx] += (gpuGlobalDataCommon[163, idx]*sum_dely_sqr - gpuGlobalDataCommon[167, idx]*sum_delx_dely)*one_by_det
    gpuGlobalDataRest[7, idx] += (gpuGlobalDataCommon[164, idx]*sum_dely_sqr - gpuGlobalDataCommon[168, idx]*sum_delx_dely)*one_by_det
    gpuGlobalDataRest[8, idx] += (gpuGlobalDataCommon[165, idx]*sum_dely_sqr - gpuGlobalDataCommon[169, idx]*sum_delx_dely)*one_by_det
    return nothing
end

function outer_dGy_pos_kernel(gpuGlobalDataCommon, gpuGlobalDataFixedPoint, gpuGlobalDataRest, idx, gpuConfigData)

    power = gpuConfigData[6]
    limiter_flag = gpuConfigData[7]

    sum_delx_sqr = 0.0
    sum_dely_sqr = 0.0
    sum_delx_dely = 0.0
    x_k = 0.0
    y_k = 0.0
    delx = 0.0
    dely = 0.0
    dels = 0.0
    deln = 0.0
    dist = 0.0
    weights = 0.0
    dels_weights = 0.0
    deln_weights = 0.0
    sum_delx_sqr = 0.0
    sum_dely_sqr = 0.0
    sum_delx_dely = 0.0

    for i in 146:173
        gpuGlobalDataCommon[i, idx] = 0.0
    end

    x_i = gpuGlobalDataFixedPoint[idx].x
    y_i = gpuGlobalDataFixedPoint[idx].y
    nx = gpuGlobalDataFixedPoint[idx].nx
    ny = gpuGlobalDataFixedPoint[idx].ny

    tx = ny
    ty = -nx

    for iter in 96:115
        conn = Int(gpuGlobalDataCommon[iter, idx])
        if conn == 0.0
            break
        end
        x_k = gpuGlobalDataFixedPoint[conn].x
        y_k = gpuGlobalDataFixedPoint[conn].y
        delx = x_k - x_i
        dely = y_k - y_i
        dels = delx*tx + dely*ty
        deln = delx*nx + dely*ny
        dist = CUDAnative.hypot(dels, deln)
        weights = CUDAnative.pow(dist, power)
        dels_weights = dels*weights
        deln_weights = deln*weights

        sum_delx_sqr = sum_delx_sqr + dels*dels_weights
        sum_dely_sqr = sum_dely_sqr + deln*deln_weights
        sum_delx_dely = sum_delx_dely + dels*deln_weights

        if limiter_flag == 1
            venkat_limiter_kernel_i(gpuGlobalDataCommon, gpuGlobalDataRest, idx, gpuConfigData, delx, dely)
            venkat_limiter_kernel_k(gpuGlobalDataCommon, gpuGlobalDataRest, conn, gpuConfigData, idx, delx, dely)
            # CUDAnative.synchronize()
            qtilde_i =  (
                            gpuGlobalDataRest[9, idx] - 0.5*gpuGlobalDataRest[29, idx]*(delx * gpuGlobalDataRest[13, idx] + dely * gpuGlobalDataRest[17, idx]),
                            gpuGlobalDataRest[10, idx] - 0.5*gpuGlobalDataRest[30, idx]*(delx * gpuGlobalDataRest[14, idx] + dely * gpuGlobalDataRest[18, idx]),
                            gpuGlobalDataRest[11, idx] - 0.5*gpuGlobalDataRest[31, idx]*(delx * gpuGlobalDataRest[15, idx] + dely * gpuGlobalDataRest[19, idx]),
                            gpuGlobalDataRest[12, idx] - 0.5*gpuGlobalDataRest[32, idx]*(delx * gpuGlobalDataRest[16, idx] + dely * gpuGlobalDataRest[20, idx])
                        )
            qtilde_k = (
                            gpuGlobalDataRest[9, conn] - 0.5*gpuGlobalDataRest[33, idx]*(delx * gpuGlobalDataRest[13, conn] + dely * gpuGlobalDataRest[17, conn]),
                            gpuGlobalDataRest[10, conn] - 0.5*gpuGlobalDataRest[34, idx]*(delx * gpuGlobalDataRest[14, conn] + dely * gpuGlobalDataRest[18, conn]),
                            gpuGlobalDataRest[11, conn] - 0.5*gpuGlobalDataRest[35, idx]*(delx * gpuGlobalDataRest[15, conn] + dely * gpuGlobalDataRest[19, conn]),
                            gpuGlobalDataRest[12, conn] - 0.5*gpuGlobalDataRest[36, idx]*(delx * gpuGlobalDataRest[16, conn] + dely * gpuGlobalDataRest[20, conn])
                        )
        end

        if limiter_flag == 2
            @cuprintf("\n Havent written the code - die \n")
        end
        qtilde_to_primitive_kernel(qtilde_i, gpuConfigData, gpuGlobalDataCommon, gpuGlobalDataRest, idx)
        flux_Gyp_kernel(nx, ny, gpuGlobalDataCommon, gpuGlobalDataRest, idx, 1)
        qtilde_to_primitive_kernel(qtilde_k, gpuConfigData, gpuGlobalDataCommon, gpuGlobalDataRest, idx)
        flux_Gyp_kernel(nx, ny, gpuGlobalDataCommon, gpuGlobalDataRest, idx, 2)
        # CUDAnative.synchronize()
        gpuGlobalDataCommon[162, idx] += (gpuGlobalDataCommon[158, idx] - gpuGlobalDataCommon[154, idx]) * dels_weights
        gpuGlobalDataCommon[166, idx] += (gpuGlobalDataCommon[158, idx] - gpuGlobalDataCommon[154, idx]) * deln_weights
        gpuGlobalDataCommon[163, idx] += (gpuGlobalDataCommon[159, idx] - gpuGlobalDataCommon[155, idx]) * dels_weights
        gpuGlobalDataCommon[167, idx] += (gpuGlobalDataCommon[159, idx] - gpuGlobalDataCommon[155, idx]) * deln_weights
        gpuGlobalDataCommon[164, idx] += (gpuGlobalDataCommon[160, idx] - gpuGlobalDataCommon[156, idx]) * dels_weights
        gpuGlobalDataCommon[168, idx] += (gpuGlobalDataCommon[160, idx] - gpuGlobalDataCommon[156, idx]) * deln_weights
        gpuGlobalDataCommon[165, idx] += (gpuGlobalDataCommon[161, idx] - gpuGlobalDataCommon[157, idx]) * dels_weights
        gpuGlobalDataCommon[169, idx] += (gpuGlobalDataCommon[161, idx] - gpuGlobalDataCommon[157, idx]) * deln_weights
    end

    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1.0 / det

    gpuGlobalDataRest[5, idx] += (gpuGlobalDataCommon[166, idx]*sum_delx_sqr - gpuGlobalDataCommon[162, idx]*sum_delx_dely)*one_by_det
    gpuGlobalDataRest[6, idx] += (gpuGlobalDataCommon[167, idx]*sum_delx_sqr - gpuGlobalDataCommon[163, idx]*sum_delx_dely)*one_by_det
    gpuGlobalDataRest[7, idx] += (gpuGlobalDataCommon[168, idx]*sum_delx_sqr - gpuGlobalDataCommon[164, idx]*sum_delx_dely)*one_by_det
    gpuGlobalDataRest[8, idx] += (gpuGlobalDataCommon[169, idx]*sum_delx_sqr - gpuGlobalDataCommon[165, idx]*sum_delx_dely)*one_by_det
    return nothing
end