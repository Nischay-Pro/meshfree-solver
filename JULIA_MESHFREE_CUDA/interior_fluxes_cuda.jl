
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
                            gpuGlobalDataRest[9, idx] - 0.5*gpuGlobalDataCommon[146, idx]*(delx * gpuGlobalDataCommon[43, idx] + dely * gpuGlobalDataCommon[47, idx]),
                            gpuGlobalDataRest[10, idx] - 0.5*gpuGlobalDataCommon[147, idx]*(delx * gpuGlobalDataCommon[44, idx] + dely * gpuGlobalDataCommon[48, idx]),
                            gpuGlobalDataRest[11, idx] - 0.5*gpuGlobalDataCommon[148, idx]*(delx * gpuGlobalDataCommon[45, idx] + dely * gpuGlobalDataCommon[49, idx]),
                            gpuGlobalDataRest[12, idx] - 0.5*gpuGlobalDataCommon[149, idx]*(delx * gpuGlobalDataCommon[46, idx] + dely * gpuGlobalDataCommon[50, idx])
                        )
            qtilde_k = (
                            gpuGlobalDataRest[9, conn] - 0.5*gpuGlobalDataCommon[150, idx]*(delx * gpuGlobalDataCommon[43, conn] + dely * gpuGlobalDataCommon[47, conn]),
                            gpuGlobalDataRest[10, conn] - 0.5*gpuGlobalDataCommon[151, idx]*(delx * gpuGlobalDataCommon[44, conn] + dely * gpuGlobalDataCommon[48, conn]),
                            gpuGlobalDataRest[11, conn] - 0.5*gpuGlobalDataCommon[152, idx]*(delx * gpuGlobalDataCommon[45, conn] + dely * gpuGlobalDataCommon[49, conn]),
                            gpuGlobalDataRest[12, conn] - 0.5*gpuGlobalDataCommon[153, idx]*(delx * gpuGlobalDataCommon[46, conn] + dely * gpuGlobalDataCommon[50, conn])
                        )
        end

        if limiter_flag == 2
            @cuprintf("\n Havent written the code - die \n")
        end
        qtilde_to_primitive_kernel(qtilde_i, gpuConfigData, gpuGlobalDataCommon, idx)
        flux_Gxp_kernel(nx, ny, gpuGlobalDataCommon, idx, 1)
        qtilde_to_primitive_kernel(qtilde_k, gpuConfigData, gpuGlobalDataCommon, idx)
        flux_Gxp_kernel(nx, ny, gpuGlobalDataCommon, idx, 2)
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
                            gpuGlobalDataRest[9, idx] - 0.5*gpuGlobalDataCommon[146, idx]*(delx * gpuGlobalDataCommon[43, idx] + dely * gpuGlobalDataCommon[47, idx]),
                            gpuGlobalDataRest[10, idx] - 0.5*gpuGlobalDataCommon[147, idx]*(delx * gpuGlobalDataCommon[44, idx] + dely * gpuGlobalDataCommon[48, idx]),
                            gpuGlobalDataRest[11, idx] - 0.5*gpuGlobalDataCommon[148, idx]*(delx * gpuGlobalDataCommon[45, idx] + dely * gpuGlobalDataCommon[49, idx]),
                            gpuGlobalDataRest[12, idx] - 0.5*gpuGlobalDataCommon[149, idx]*(delx * gpuGlobalDataCommon[46, idx] + dely * gpuGlobalDataCommon[50, idx])
                        )
            qtilde_k = (
                            gpuGlobalDataRest[9, conn] - 0.5*gpuGlobalDataCommon[150, idx]*(delx * gpuGlobalDataCommon[43, conn] + dely * gpuGlobalDataCommon[47, conn]),
                            gpuGlobalDataRest[10, conn] - 0.5*gpuGlobalDataCommon[151, idx]*(delx * gpuGlobalDataCommon[44, conn] + dely * gpuGlobalDataCommon[48, conn]),
                            gpuGlobalDataRest[11, conn] - 0.5*gpuGlobalDataCommon[152, idx]*(delx * gpuGlobalDataCommon[45, conn] + dely * gpuGlobalDataCommon[49, conn]),
                            gpuGlobalDataRest[12, conn] - 0.5*gpuGlobalDataCommon[153, idx]*(delx * gpuGlobalDataCommon[46, conn] + dely * gpuGlobalDataCommon[50, conn])
                        )
        end

        if limiter_flag == 2
            @cuprintf("\n Havent written the code - die \n")
        end
        qtilde_to_primitive_kernel(qtilde_i, gpuConfigData, gpuGlobalDataCommon, idx)
        flux_Gxn_kernel(nx, ny, gpuGlobalDataCommon, idx, 1)
        qtilde_to_primitive_kernel(qtilde_k, gpuConfigData, gpuGlobalDataCommon, idx)
        flux_Gxn_kernel(nx, ny, gpuGlobalDataCommon, idx, 2)
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
    # if idx == 1
    #     @cuprintf("\n===Gyn===")
    #     @cuprintf("\n %f", sum_delx_sqr)
    #     @cuprintf("\n %f", sum_dely_sqr)
    #     @cuprintf("\n %f", sum_delx_dely)
    #     @cuprintf("\n %f", det)
    #     @cuprintf("\n %.17f %.17f %.17f %.17f", gpuGlobalDataCommon[162, idx],gpuGlobalDataCommon[163, idx],gpuGlobalDataCommon[164, idx],gpuGlobalDataCommon[165, idx])
    #     @cuprintf("\n %.17f %.17f %.17f %.17f", gpuGlobalDataCommon[166, idx],gpuGlobalDataCommon[167, idx],gpuGlobalDataCommon[168, idx],gpuGlobalDataCommon[169, idx])
    # end
    gpuGlobalDataRest[5, idx] += (gpuGlobalDataCommon[162, idx]*sum_dely_sqr - gpuGlobalDataCommon[166, idx]*sum_delx_dely)*one_by_det
    gpuGlobalDataRest[6, idx] += (gpuGlobalDataCommon[163, idx]*sum_dely_sqr - gpuGlobalDataCommon[167, idx]*sum_delx_dely)*one_by_det
    gpuGlobalDataRest[7, idx] += (gpuGlobalDataCommon[164, idx]*sum_dely_sqr - gpuGlobalDataCommon[168, idx]*sum_delx_dely)*one_by_det
    gpuGlobalDataRest[8, idx] += (gpuGlobalDataCommon[165, idx]*sum_dely_sqr - gpuGlobalDataCommon[169, idx]*sum_delx_dely)*one_by_det
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
                            gpuGlobalDataRest[9, idx] - 0.5*gpuGlobalDataCommon[146, idx]*(delx * gpuGlobalDataCommon[43, idx] + dely * gpuGlobalDataCommon[47, idx]),
                            gpuGlobalDataRest[10, idx] - 0.5*gpuGlobalDataCommon[147, idx]*(delx * gpuGlobalDataCommon[44, idx] + dely * gpuGlobalDataCommon[48, idx]),
                            gpuGlobalDataRest[11, idx] - 0.5*gpuGlobalDataCommon[148, idx]*(delx * gpuGlobalDataCommon[45, idx] + dely * gpuGlobalDataCommon[49, idx]),
                            gpuGlobalDataRest[12, idx] - 0.5*gpuGlobalDataCommon[149, idx]*(delx * gpuGlobalDataCommon[46, idx] + dely * gpuGlobalDataCommon[50, idx])
                        )
            qtilde_k = (
                            gpuGlobalDataRest[9, conn] - 0.5*gpuGlobalDataCommon[150, idx]*(delx * gpuGlobalDataCommon[43, conn] + dely * gpuGlobalDataCommon[47, conn]),
                            gpuGlobalDataRest[10, conn] - 0.5*gpuGlobalDataCommon[151, idx]*(delx * gpuGlobalDataCommon[44, conn] + dely * gpuGlobalDataCommon[48, conn]),
                            gpuGlobalDataRest[11, conn] - 0.5*gpuGlobalDataCommon[152, idx]*(delx * gpuGlobalDataCommon[45, conn] + dely * gpuGlobalDataCommon[49, conn]),
                            gpuGlobalDataRest[12, conn] - 0.5*gpuGlobalDataCommon[153, idx]*(delx * gpuGlobalDataCommon[46, conn] + dely * gpuGlobalDataCommon[50, conn])
                        )
        end

        if limiter_flag == 2
            @cuprintf("\n Havent written the code - die \n")
        end
        qtilde_to_primitive_kernel(qtilde_i, gpuConfigData, gpuGlobalDataCommon, idx)
        flux_Gyp_kernel(nx, ny, gpuGlobalDataCommon, idx, 1)
        qtilde_to_primitive_kernel(qtilde_k, gpuConfigData, gpuGlobalDataCommon, idx)
        flux_Gyp_kernel(nx, ny, gpuGlobalDataCommon, idx, 2)
        # CUDAnative.synchronize()
        gpuGlobalDataCommon[162, idx] += (gpuGlobalDataCommon[158, idx] - gpuGlobalDataCommon[154, idx]) * dels_weights
        gpuGlobalDataCommon[166, idx] += (gpuGlobalDataCommon[158, idx] - gpuGlobalDataCommon[154, idx]) * deln_weights
        gpuGlobalDataCommon[163, idx] += (gpuGlobalDataCommon[159, idx] - gpuGlobalDataCommon[155, idx]) * dels_weights
        gpuGlobalDataCommon[167, idx] += (gpuGlobalDataCommon[159, idx] - gpuGlobalDataCommon[155, idx]) * deln_weights
        gpuGlobalDataCommon[164, idx] += (gpuGlobalDataCommon[160, idx] - gpuGlobalDataCommon[156, idx]) * dels_weights
        gpuGlobalDataCommon[168, idx] += (gpuGlobalDataCommon[160, idx] - gpuGlobalDataCommon[156, idx]) * deln_weights
        gpuGlobalDataCommon[165, idx] += (gpuGlobalDataCommon[161, idx] - gpuGlobalDataCommon[157, idx]) * dels_weights
        gpuGlobalDataCommon[169, idx] += (gpuGlobalDataCommon[161, idx] - gpuGlobalDataCommon[157, idx]) * deln_weights
        # if idx == 200
        #     @cuprintf("\n %d", conn)
        #     @cuprintf("\n %.17f %.17f %.17f %.17f", gpuGlobalDataCommon[170, idx], gpuGlobalDataCommon[171, idx], gpuGlobalDataCommon[172, idx], gpuGlobalDataCommon[173, idx])
        #     @cuprintf("\n %.17f %.17f %.17f %.17f", gpuGlobalDataCommon[154, idx], gpuGlobalDataCommon[155, idx], gpuGlobalDataCommon[156, idx], gpuGlobalDataCommon[157, idx])
        #     @cuprintf("\n %.17f %.17f %.17f %.17f", gpuGlobalDataCommon[158, idx], gpuGlobalDataCommon[159, idx], gpuGlobalDataCommon[160, idx], gpuGlobalDataCommon[161, idx])
        # end
    end

    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1.0 / det
    gpuGlobalDataRest[5, idx] += (gpuGlobalDataCommon[166, idx]*sum_delx_sqr - gpuGlobalDataCommon[162, idx]*sum_delx_dely)*one_by_det
    gpuGlobalDataRest[6, idx] += (gpuGlobalDataCommon[167, idx]*sum_delx_sqr - gpuGlobalDataCommon[163, idx]*sum_delx_dely)*one_by_det
    gpuGlobalDataRest[7, idx] += (gpuGlobalDataCommon[168, idx]*sum_delx_sqr - gpuGlobalDataCommon[164, idx]*sum_delx_dely)*one_by_det
    gpuGlobalDataRest[8, idx] += (gpuGlobalDataCommon[169, idx]*sum_delx_sqr - gpuGlobalDataCommon[165, idx]*sum_delx_dely)*one_by_det
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
                            gpuGlobalDataRest[9, idx] - 0.5*gpuGlobalDataCommon[146, idx]*(delx * gpuGlobalDataCommon[43, idx] + dely * gpuGlobalDataCommon[47, idx]),
                            gpuGlobalDataRest[10, idx] - 0.5*gpuGlobalDataCommon[147, idx]*(delx * gpuGlobalDataCommon[44, idx] + dely * gpuGlobalDataCommon[48, idx]),
                            gpuGlobalDataRest[11, idx] - 0.5*gpuGlobalDataCommon[148, idx]*(delx * gpuGlobalDataCommon[45, idx] + dely * gpuGlobalDataCommon[49, idx]),
                            gpuGlobalDataRest[12, idx] - 0.5*gpuGlobalDataCommon[149, idx]*(delx * gpuGlobalDataCommon[46, idx] + dely * gpuGlobalDataCommon[50, idx])
                        )
            qtilde_k = (
                            gpuGlobalDataRest[9, conn] - 0.5*gpuGlobalDataCommon[150, idx]*(delx * gpuGlobalDataCommon[43, conn] + dely * gpuGlobalDataCommon[47, conn]),
                            gpuGlobalDataRest[10, conn] - 0.5*gpuGlobalDataCommon[151, idx]*(delx * gpuGlobalDataCommon[44, conn] + dely * gpuGlobalDataCommon[48, conn]),
                            gpuGlobalDataRest[11, conn] - 0.5*gpuGlobalDataCommon[152, idx]*(delx * gpuGlobalDataCommon[45, conn] + dely * gpuGlobalDataCommon[49, conn]),
                            gpuGlobalDataRest[12, conn] - 0.5*gpuGlobalDataCommon[153, idx]*(delx * gpuGlobalDataCommon[46, conn] + dely * gpuGlobalDataCommon[50, conn])
                        )
        end
        if limiter_flag == 2
            @cuprintf("\n Havent written the code - die \n")
        end
        qtilde_to_primitive_kernel(qtilde_i, gpuConfigData, gpuGlobalDataCommon, idx)
        flux_Gyn_kernel(nx, ny, gpuGlobalDataCommon, idx, 1)
        qtilde_to_primitive_kernel(qtilde_k, gpuConfigData, gpuGlobalDataCommon, idx)
        flux_Gyn_kernel(nx, ny, gpuGlobalDataCommon, idx, 2)
        # CUDAnative.synchronize()
        gpuGlobalDataCommon[162, idx] += (gpuGlobalDataCommon[158, idx] - gpuGlobalDataCommon[154, idx]) * dels_weights
        gpuGlobalDataCommon[166, idx] += (gpuGlobalDataCommon[158, idx] - gpuGlobalDataCommon[154, idx]) * deln_weights
        gpuGlobalDataCommon[163, idx] += (gpuGlobalDataCommon[159, idx] - gpuGlobalDataCommon[155, idx]) * dels_weights
        gpuGlobalDataCommon[167, idx] += (gpuGlobalDataCommon[159, idx] - gpuGlobalDataCommon[155, idx]) * deln_weights
        gpuGlobalDataCommon[164, idx] += (gpuGlobalDataCommon[160, idx] - gpuGlobalDataCommon[156, idx]) * dels_weights
        gpuGlobalDataCommon[168, idx] += (gpuGlobalDataCommon[160, idx] - gpuGlobalDataCommon[156, idx]) * deln_weights
        gpuGlobalDataCommon[165, idx] += (gpuGlobalDataCommon[161, idx] - gpuGlobalDataCommon[157, idx]) * dels_weights
        gpuGlobalDataCommon[169, idx] += (gpuGlobalDataCommon[161, idx] - gpuGlobalDataCommon[157, idx]) * deln_weights
        # if idx == 1
        #     @cuprintf("\n %d", conn)
        #     @cuprintf("\n %.17f %.17f %.17f %.17f", gpuGlobalDataCommon[170, idx], gpuGlobalDataCommon[171, idx], gpuGlobalDataCommon[172, idx], gpuGlobalDataCommon[173, idx])
        #     @cuprintf("\n %.17f %.17f %.17f %.17f", gpuGlobalDataCommon[154, idx], gpuGlobalDataCommon[155, idx], gpuGlobalDataCommon[156, idx], gpuGlobalDataCommon[157, idx])
        #     @cuprintf("\n %.17f %.17f %.17f %.17f", gpuGlobalDataCommon[158, idx], gpuGlobalDataCommon[159, idx], gpuGlobalDataCommon[160, idx], gpuGlobalDataCommon[161, idx])
        # end
    end

    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1.0 / det
    gpuGlobalDataRest[5, idx] += (gpuGlobalDataCommon[166, idx]*sum_delx_sqr - gpuGlobalDataCommon[162, idx]*sum_delx_dely)*one_by_det
    gpuGlobalDataRest[6, idx] += (gpuGlobalDataCommon[167, idx]*sum_delx_sqr - gpuGlobalDataCommon[163, idx]*sum_delx_dely)*one_by_det
    gpuGlobalDataRest[7, idx] += (gpuGlobalDataCommon[168, idx]*sum_delx_sqr - gpuGlobalDataCommon[164, idx]*sum_delx_dely)*one_by_det
    gpuGlobalDataRest[8, idx] += (gpuGlobalDataCommon[169, idx]*sum_delx_sqr - gpuGlobalDataCommon[165, idx]*sum_delx_dely)*one_by_det
    # if idx ==1
    #     @cuprintf("\n %f %f %f %f", gpuGlobalDataRest[5, idx],gpuGlobalDataRest[6, idx],gpuGlobalDataRest[7, idx],gpuGlobalDataRest[8, idx])
    # end
    return nothing
end
