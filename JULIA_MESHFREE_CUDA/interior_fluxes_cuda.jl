
function interior_dGx_pos_kernel(gpuGlobalDataCommon, gpuGlobalDataFixedPoint, gpuGlobalDataRest, idx, gpuConfigData)
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

    for i in 154:173
        gpuGlobalDataCommon[i, idx] = 0.0
    end
    for i in 37:56
        gpuGlobalDataRest[i, idx] = 0.0
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
        flux_Gxp_kernel(nx, ny, gpuGlobalDataCommon, gpuGlobalDataRest, idx, 1)
        qtilde_to_primitive_kernel(qtilde_k, gpuConfigData, gpuGlobalDataCommon, gpuGlobalDataRest, idx)
        flux_Gxp_kernel(nx, ny, gpuGlobalDataCommon, gpuGlobalDataRest, idx, 2)
        # CUDAnative.synchronize()
        gpuGlobalDataRest[45, idx] += (gpuGlobalDataRest[41, idx] - gpuGlobalDataRest[37, idx]) * dels_weights
        gpuGlobalDataRest[49, idx] += (gpuGlobalDataRest[41, idx] - gpuGlobalDataRest[37, idx]) * deln_weights
        gpuGlobalDataRest[46, idx] += (gpuGlobalDataRest[42, idx] - gpuGlobalDataRest[38, idx]) * dels_weights
        gpuGlobalDataRest[50, idx] += (gpuGlobalDataRest[42, idx] - gpuGlobalDataRest[38, idx]) * deln_weights
        gpuGlobalDataRest[47, idx] += (gpuGlobalDataRest[43, idx] - gpuGlobalDataRest[39, idx]) * dels_weights
        gpuGlobalDataRest[51, idx] += (gpuGlobalDataRest[43, idx] - gpuGlobalDataRest[39, idx]) * deln_weights
        gpuGlobalDataRest[48, idx] += (gpuGlobalDataRest[44, idx] - gpuGlobalDataRest[40, idx]) * dels_weights
        gpuGlobalDataRest[52, idx] += (gpuGlobalDataRest[44, idx] - gpuGlobalDataRest[40, idx]) * deln_weights
    end

    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1.0 / det

    gpuGlobalDataRest[5, idx] += (gpuGlobalDataRest[45, idx]*sum_dely_sqr - gpuGlobalDataRest[49, idx]*sum_delx_dely)*one_by_det
    gpuGlobalDataRest[6, idx] += (gpuGlobalDataRest[46, idx]*sum_dely_sqr - gpuGlobalDataRest[50, idx]*sum_delx_dely)*one_by_det
    gpuGlobalDataRest[7, idx] += (gpuGlobalDataRest[47, idx]*sum_dely_sqr - gpuGlobalDataRest[51, idx]*sum_delx_dely)*one_by_det
    gpuGlobalDataRest[8, idx] += (gpuGlobalDataRest[48, idx]*sum_dely_sqr - gpuGlobalDataRest[52, idx]*sum_delx_dely)*one_by_det
    # if idx == 1
    #     @cuprintf("\n %f %f %f %f", gpuGlobalDataRest[5, idx],gpuGlobalDataRest[6, idx],gpuGlobalDataRest[7, idx],gpuGlobalDataRest[8, idx])
    # end
    return nothing
end

function interior_dGx_neg_kernel(gpuGlobalDataCommon, gpuGlobalDataFixedPoint, gpuGlobalDataRest, idx, gpuConfigData)
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

    for i in 154:173
        gpuGlobalDataCommon[i, idx] = 0.0
    end
    for i in 37:56
        gpuGlobalDataRest[i, idx] = 0.0
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
        flux_Gxn_kernel(nx, ny, gpuGlobalDataCommon, gpuGlobalDataRest, idx, 1)
        qtilde_to_primitive_kernel(qtilde_k, gpuConfigData, gpuGlobalDataCommon, gpuGlobalDataRest, idx)
        flux_Gxn_kernel(nx, ny, gpuGlobalDataCommon, gpuGlobalDataRest, idx, 2)
        # CUDAnative.synchronize()
        gpuGlobalDataRest[45, idx] += (gpuGlobalDataRest[41, idx] - gpuGlobalDataRest[37, idx]) * dels_weights
        gpuGlobalDataRest[49, idx] += (gpuGlobalDataRest[41, idx] - gpuGlobalDataRest[37, idx]) * deln_weights
        gpuGlobalDataRest[46, idx] += (gpuGlobalDataRest[42, idx] - gpuGlobalDataRest[38, idx]) * dels_weights
        gpuGlobalDataRest[50, idx] += (gpuGlobalDataRest[42, idx] - gpuGlobalDataRest[38, idx]) * deln_weights
        gpuGlobalDataRest[47, idx] += (gpuGlobalDataRest[43, idx] - gpuGlobalDataRest[39, idx]) * dels_weights
        gpuGlobalDataRest[51, idx] += (gpuGlobalDataRest[43, idx] - gpuGlobalDataRest[39, idx]) * deln_weights
        gpuGlobalDataRest[48, idx] += (gpuGlobalDataRest[44, idx] - gpuGlobalDataRest[40, idx]) * dels_weights
        gpuGlobalDataRest[52, idx] += (gpuGlobalDataRest[44, idx] - gpuGlobalDataRest[40, idx]) * deln_weights
    end

    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1.0 / det
    # if idx == 1
    #     @cuprintf("\n===Gyn===")
    #     @cuprintf("\n %f", sum_delx_sqr)
    #     @cuprintf("\n %f", sum_dely_sqr)
    #     @cuprintf("\n %f", sum_delx_dely)
    #     @cuprintf("\n %f", det)
    #     @cuprintf("\n %.17f %.17f %.17f %.17f", gpuGlobalDataRest[45, idx],gpuGlobalDataRest[46, idx],gpuGlobalDataRest[47, idx],gpuGlobalDataRest[48, idx])
    #     @cuprintf("\n %.17f %.17f %.17f %.17f", gpuGlobalDataRest[49, idx],gpuGlobalDataRest[50, idx],gpuGlobalDataRest[51, idx],gpuGlobalDataRest[52, idx])
    # end
    gpuGlobalDataRest[5, idx] += (gpuGlobalDataRest[45, idx]*sum_dely_sqr - gpuGlobalDataRest[49, idx]*sum_delx_dely)*one_by_det
    gpuGlobalDataRest[6, idx] += (gpuGlobalDataRest[46, idx]*sum_dely_sqr - gpuGlobalDataRest[50, idx]*sum_delx_dely)*one_by_det
    gpuGlobalDataRest[7, idx] += (gpuGlobalDataRest[47, idx]*sum_dely_sqr - gpuGlobalDataRest[51, idx]*sum_delx_dely)*one_by_det
    gpuGlobalDataRest[8, idx] += (gpuGlobalDataRest[48, idx]*sum_dely_sqr - gpuGlobalDataRest[52, idx]*sum_delx_dely)*one_by_det
    # if idx == 1
    #     @cuprintf("\n %f %f %f %f", gpuGlobalDataRest[5, idx],gpuGlobalDataRest[6, idx],gpuGlobalDataRest[7, idx],gpuGlobalDataRest[8, idx])
    # end
    return nothing
end

function interior_dGy_pos_kernel(gpuGlobalDataCommon, gpuGlobalDataFixedPoint, gpuGlobalDataRest, idx, gpuConfigData)
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

    for i in 154:173
        gpuGlobalDataCommon[i, idx] = 0.0
    end
    for i in 37:56
        gpuGlobalDataRest[i, idx] = 0.0
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
        gpuGlobalDataRest[45, idx] += (gpuGlobalDataRest[41, idx] - gpuGlobalDataRest[37, idx]) * dels_weights
        gpuGlobalDataRest[49, idx] += (gpuGlobalDataRest[41, idx] - gpuGlobalDataRest[37, idx]) * deln_weights
        gpuGlobalDataRest[46, idx] += (gpuGlobalDataRest[42, idx] - gpuGlobalDataRest[38, idx]) * dels_weights
        gpuGlobalDataRest[50, idx] += (gpuGlobalDataRest[42, idx] - gpuGlobalDataRest[38, idx]) * deln_weights
        gpuGlobalDataRest[47, idx] += (gpuGlobalDataRest[43, idx] - gpuGlobalDataRest[39, idx]) * dels_weights
        gpuGlobalDataRest[51, idx] += (gpuGlobalDataRest[43, idx] - gpuGlobalDataRest[39, idx]) * deln_weights
        gpuGlobalDataRest[48, idx] += (gpuGlobalDataRest[44, idx] - gpuGlobalDataRest[40, idx]) * dels_weights
        gpuGlobalDataRest[52, idx] += (gpuGlobalDataRest[44, idx] - gpuGlobalDataRest[40, idx]) * deln_weights
        # if idx == 200
        #     @cuprintf("\n %d", conn)
        #     @cuprintf("\n %.17f %.17f %.17f %.17f", gpuGlobalDataRest[53, idx], gpuGlobalDataRest[54, idx], gpuGlobalDataRest[55, idx], gpuGlobalDataRest[56, idx])
        #     @cuprintf("\n %.17f %.17f %.17f %.17f", gpuGlobalDataRest[37, idx], gpuGlobalDataRest[38, idx], gpuGlobalDataRest[39, idx], gpuGlobalDataRest[40, idx])
        #     @cuprintf("\n %.17f %.17f %.17f %.17f", gpuGlobalDataRest[41, idx], gpuGlobalDataRest[42, idx], gpuGlobalDataRest[43, idx], gpuGlobalDataRest[44, idx])
        # end
    end

    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1.0 / det
    gpuGlobalDataRest[5, idx] += (gpuGlobalDataRest[49, idx]*sum_delx_sqr - gpuGlobalDataRest[45, idx]*sum_delx_dely)*one_by_det
    gpuGlobalDataRest[6, idx] += (gpuGlobalDataRest[50, idx]*sum_delx_sqr - gpuGlobalDataRest[46, idx]*sum_delx_dely)*one_by_det
    gpuGlobalDataRest[7, idx] += (gpuGlobalDataRest[51, idx]*sum_delx_sqr - gpuGlobalDataRest[47, idx]*sum_delx_dely)*one_by_det
    gpuGlobalDataRest[8, idx] += (gpuGlobalDataRest[52, idx]*sum_delx_sqr - gpuGlobalDataRest[48, idx]*sum_delx_dely)*one_by_det
    # if idx ==1
    #     @cuprintf("\n %f %f %f %f", gpuGlobalDataRest[5, idx],gpuGlobalDataRest[6, idx],gpuGlobalDataRest[7, idx],gpuGlobalDataRest[8, idx])
    # end
    return nothing
end

function interior_dGy_neg_kernel(gpuGlobalDataCommon, gpuGlobalDataFixedPoint, gpuGlobalDataRest, idx, gpuConfigData)
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

    for i in 154:173
        gpuGlobalDataCommon[i, idx] = 0.0
    end
    for i in 37:56
        gpuGlobalDataRest[i, idx] = 0.0
    end

    x_i = gpuGlobalDataFixedPoint[idx].x
    y_i = gpuGlobalDataFixedPoint[idx].y
    nx = gpuGlobalDataFixedPoint[idx].nx
    ny = gpuGlobalDataFixedPoint[idx].ny

    tx = ny
    ty = -nx

    for iter in 116:135
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
        flux_Gyn_kernel(nx, ny, gpuGlobalDataCommon, gpuGlobalDataRest, idx, 1)
        qtilde_to_primitive_kernel(qtilde_k, gpuConfigData, gpuGlobalDataCommon, gpuGlobalDataRest, idx)
        flux_Gyn_kernel(nx, ny, gpuGlobalDataCommon, gpuGlobalDataRest, idx, 2)
        # CUDAnative.synchronize()
        gpuGlobalDataRest[45, idx] += (gpuGlobalDataRest[41, idx] - gpuGlobalDataRest[37, idx]) * dels_weights
        gpuGlobalDataRest[49, idx] += (gpuGlobalDataRest[41, idx] - gpuGlobalDataRest[37, idx]) * deln_weights
        gpuGlobalDataRest[46, idx] += (gpuGlobalDataRest[42, idx] - gpuGlobalDataRest[38, idx]) * dels_weights
        gpuGlobalDataRest[50, idx] += (gpuGlobalDataRest[42, idx] - gpuGlobalDataRest[38, idx]) * deln_weights
        gpuGlobalDataRest[47, idx] += (gpuGlobalDataRest[43, idx] - gpuGlobalDataRest[39, idx]) * dels_weights
        gpuGlobalDataRest[51, idx] += (gpuGlobalDataRest[43, idx] - gpuGlobalDataRest[39, idx]) * deln_weights
        gpuGlobalDataRest[48, idx] += (gpuGlobalDataRest[44, idx] - gpuGlobalDataRest[40, idx]) * dels_weights
        gpuGlobalDataRest[52, idx] += (gpuGlobalDataRest[44, idx] - gpuGlobalDataRest[40, idx]) * deln_weights
        # if idx == 1
        #     @cuprintf("\n %d", conn)
        #     @cuprintf("\n %.17f %.17f %.17f %.17f", gpuGlobalDataRest[53, idx], gpuGlobalDataRest[54, idx], gpuGlobalDataRest[55, idx], gpuGlobalDataRest[56, idx])
        #     @cuprintf("\n %.17f %.17f %.17f %.17f", gpuGlobalDataRest[37, idx], gpuGlobalDataRest[38, idx], gpuGlobalDataRest[39, idx], gpuGlobalDataRest[40, idx])
        #     @cuprintf("\n %.17f %.17f %.17f %.17f", gpuGlobalDataRest[41, idx], gpuGlobalDataRest[42, idx], gpuGlobalDataRest[43, idx], gpuGlobalDataRest[44, idx])
        # end
    end

    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1.0 / det
    gpuGlobalDataRest[5, idx] += (gpuGlobalDataRest[49, idx]*sum_delx_sqr - gpuGlobalDataRest[45, idx]*sum_delx_dely)*one_by_det
    gpuGlobalDataRest[6, idx] += (gpuGlobalDataRest[50, idx]*sum_delx_sqr - gpuGlobalDataRest[46, idx]*sum_delx_dely)*one_by_det
    gpuGlobalDataRest[7, idx] += (gpuGlobalDataRest[51, idx]*sum_delx_sqr - gpuGlobalDataRest[47, idx]*sum_delx_dely)*one_by_det
    gpuGlobalDataRest[8, idx] += (gpuGlobalDataRest[52, idx]*sum_delx_sqr - gpuGlobalDataRest[48, idx]*sum_delx_dely)*one_by_det
    # if idx ==1
    #     @cuprintf("\n %f %f %f %f", gpuGlobalDataRest[5, idx],gpuGlobalDataRest[6, idx],gpuGlobalDataRest[7, idx],gpuGlobalDataRest[8, idx])
    # end
    return nothing
end
