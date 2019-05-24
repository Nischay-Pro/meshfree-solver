
function interior_dGx_pos_kernel(gpuGlobalDataCommon, idx, gpuConfigData, Gxp, phi_i, phi_k, G_i, G_k, sum_delx_delf, sum_dely_delf, result)

    power = gpuConfigData[6]
    limiter_flag = gpuConfigData[7]

    fill!(phi_i, 0.0)
    fill!(phi_i, 0.0)
    fill!(G_i, 0.0)
    fill!(G_k, 0.0)

    sum_delx_sqr = 0.0
    sum_dely_sqr = 0.0
    sum_delx_dely = 0.0

    fill!(sum_delx_delf,0.0)
    fill!(sum_dely_delf,0.0)
    qtilde_i = (0,0,0,0)
    qtilde_k = (0,0,0,0)

    x_i = gpuGlobalDataCommon[2, idx]
    y_i = gpuGlobalDataCommon[3, idx]
    nx = gpuGlobalDataCommon[29, idx]
    ny = gpuGlobalDataCommon[30, idx]

    tx = ny
    ty = -nx

    for iter in 56:75
        conn = Int(gpuGlobalDataCommon[iter, idx])
        if conn == 0.0
            break
        end

        x_k = gpuGlobalDataCommon[2, conn]
        y_k = gpuGlobalDataCommon[3, conn]
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

        qtilde_i =  (
                        gpuGlobalDataCommon[39, idx] - 0.5*(delx * gpuGlobalDataCommon[43, idx] + dely * gpuGlobalDataCommon[47, idx]),
                        gpuGlobalDataCommon[40, idx] - 0.5*(delx * gpuGlobalDataCommon[44, idx] + dely * gpuGlobalDataCommon[48, idx]),
                        gpuGlobalDataCommon[41, idx] - 0.5*(delx * gpuGlobalDataCommon[45, idx] + dely * gpuGlobalDataCommon[49, idx]),
                        gpuGlobalDataCommon[42, idx] - 0.5*(delx * gpuGlobalDataCommon[46, idx] + dely * gpuGlobalDataCommon[50, idx])
                    )
        qtilde_k = (
                        gpuGlobalDataCommon[39, conn] - 0.5*(delx * gpuGlobalDataCommon[43, conn] + dely * gpuGlobalDataCommon[47, conn]),
                        gpuGlobalDataCommon[40, conn] - 0.5*(delx * gpuGlobalDataCommon[44, conn] + dely * gpuGlobalDataCommon[48, conn]),
                        gpuGlobalDataCommon[41, conn] - 0.5*(delx * gpuGlobalDataCommon[45, conn] + dely * gpuGlobalDataCommon[49, conn]),
                        gpuGlobalDataCommon[42, conn] - 0.5*(delx * gpuGlobalDataCommon[46, conn] + dely * gpuGlobalDataCommon[50, conn])
                    )

        if limiter_flag == 1
            venkat_limiter_kernel(qtilde_i, gpuGlobalDataCommon, idx, gpuConfigData, phi_i)
            venkat_limiter_kernel(qtilde_k, gpuGlobalDataCommon, conn, gpuConfigData, phi_k)
            # CUDAnative.synchronize()
            qtilde_i =  (
                            gpuGlobalDataCommon[39, idx] - 0.5*phi_i[1]*(delx * gpuGlobalDataCommon[43, idx] + dely * gpuGlobalDataCommon[47, idx]),
                            gpuGlobalDataCommon[40, idx] - 0.5*phi_i[2]*(delx * gpuGlobalDataCommon[44, idx] + dely * gpuGlobalDataCommon[48, idx]),
                            gpuGlobalDataCommon[41, idx] - 0.5*phi_i[3]*(delx * gpuGlobalDataCommon[45, idx] + dely * gpuGlobalDataCommon[49, idx]),
                            gpuGlobalDataCommon[42, idx] - 0.5*phi_i[4]*(delx * gpuGlobalDataCommon[46, idx] + dely * gpuGlobalDataCommon[50, idx])
                        )
            qtilde_k = (
                            gpuGlobalDataCommon[39, conn] - 0.5*phi_k[1]*(delx * gpuGlobalDataCommon[43, conn] + dely * gpuGlobalDataCommon[47, conn]),
                            gpuGlobalDataCommon[40, conn] - 0.5*phi_k[2]*(delx * gpuGlobalDataCommon[44, conn] + dely * gpuGlobalDataCommon[48, conn]),
                            gpuGlobalDataCommon[41, conn] - 0.5*phi_k[3]*(delx * gpuGlobalDataCommon[45, conn] + dely * gpuGlobalDataCommon[49, conn]),
                            gpuGlobalDataCommon[42, conn] - 0.5*phi_k[4]*(delx * gpuGlobalDataCommon[46, conn] + dely * gpuGlobalDataCommon[50, conn])
                        )
        end

        if limiter_flag == 2
            @cuprintf("\n Havent written the code - die \n")
        end

        qtilde_to_primitive_kernel(qtilde_i, gpuConfigData, result)
        flux_Gxp_kernel(nx, ny, result[1], result[2], result[3], result[4], G_i)
        qtilde_to_primitive_kernel(qtilde_k, gpuConfigData, result)
        flux_Gxp_kernel(nx, ny, result[1], result[2], result[3], result[4], G_k)
        # CUDAnative.synchronize()
        for i in 1:4
            sum_delx_delf[i] += (G_k[i] - G_i[i]) * dels_weights
            sum_dely_delf[i] += (G_k[i] - G_i[i]) * deln_weights
        end
    end

    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1.0 / det
    for i in 1:4
        Gxp[i] = (sum_delx_delf[i]*sum_dely_sqr - sum_dely_delf[i]*sum_delx_dely)*one_by_det
    end
    return nothing
end

function interior_dGx_neg_kernel(gpuGlobalDataCommon, idx, gpuConfigData, Gxn, phi_i, phi_k, G_i, G_k, sum_delx_delf, sum_dely_delf, result)

    power = gpuConfigData[6]
    limiter_flag = gpuConfigData[7]

    fill!(phi_i, 0.0)
    fill!(phi_i, 0.0)
    fill!(G_i, 0.0)
    fill!(G_k, 0.0)

    sum_delx_sqr = 0.0
    sum_dely_sqr = 0.0
    sum_delx_dely = 0.0

    fill!(sum_delx_delf,0.0)
    fill!(sum_dely_delf,0.0)
    qtilde_i = (0,0,0,0)
    qtilde_k = (0,0,0,0)

    x_i = gpuGlobalDataCommon[2, idx]
    y_i = gpuGlobalDataCommon[3, idx]
    nx = gpuGlobalDataCommon[29, idx]
    ny = gpuGlobalDataCommon[30, idx]

    tx = ny
    ty = -nx

    for iter in 76:95
        conn = Int(gpuGlobalDataCommon[iter, idx])
        if conn == 0.0
            break
        end

        x_k = gpuGlobalDataCommon[2, conn]
        y_k = gpuGlobalDataCommon[3, conn]
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

        qtilde_i =  (
                        gpuGlobalDataCommon[39, idx] - 0.5*(delx * gpuGlobalDataCommon[43, idx] + dely * gpuGlobalDataCommon[47, idx]),
                        gpuGlobalDataCommon[40, idx] - 0.5*(delx * gpuGlobalDataCommon[44, idx] + dely * gpuGlobalDataCommon[48, idx]),
                        gpuGlobalDataCommon[41, idx] - 0.5*(delx * gpuGlobalDataCommon[45, idx] + dely * gpuGlobalDataCommon[49, idx]),
                        gpuGlobalDataCommon[42, idx] - 0.5*(delx * gpuGlobalDataCommon[46, idx] + dely * gpuGlobalDataCommon[50, idx])
                    )
        qtilde_k = (
                        gpuGlobalDataCommon[39, conn] - 0.5*(delx * gpuGlobalDataCommon[43, conn] + dely * gpuGlobalDataCommon[47, conn]),
                        gpuGlobalDataCommon[40, conn] - 0.5*(delx * gpuGlobalDataCommon[44, conn] + dely * gpuGlobalDataCommon[48, conn]),
                        gpuGlobalDataCommon[41, conn] - 0.5*(delx * gpuGlobalDataCommon[45, conn] + dely * gpuGlobalDataCommon[49, conn]),
                        gpuGlobalDataCommon[42, conn] - 0.5*(delx * gpuGlobalDataCommon[46, conn] + dely * gpuGlobalDataCommon[50, conn])
                    )

        if limiter_flag == 1
            venkat_limiter_kernel(qtilde_i, gpuGlobalDataCommon, idx, gpuConfigData, phi_i)
            venkat_limiter_kernel(qtilde_k, gpuGlobalDataCommon, conn, gpuConfigData, phi_k)
            # CUDAnative.synchronize()
            qtilde_i =  (
                            gpuGlobalDataCommon[39, idx] - 0.5*phi_i[1]*(delx * gpuGlobalDataCommon[43, idx] + dely * gpuGlobalDataCommon[47, idx]),
                            gpuGlobalDataCommon[40, idx] - 0.5*phi_i[2]*(delx * gpuGlobalDataCommon[44, idx] + dely * gpuGlobalDataCommon[48, idx]),
                            gpuGlobalDataCommon[41, idx] - 0.5*phi_i[3]*(delx * gpuGlobalDataCommon[45, idx] + dely * gpuGlobalDataCommon[49, idx]),
                            gpuGlobalDataCommon[42, idx] - 0.5*phi_i[4]*(delx * gpuGlobalDataCommon[46, idx] + dely * gpuGlobalDataCommon[50, idx])
                        )
            qtilde_k = (
                            gpuGlobalDataCommon[39, conn] - 0.5*phi_k[1]*(delx * gpuGlobalDataCommon[43, conn] + dely * gpuGlobalDataCommon[47, conn]),
                            gpuGlobalDataCommon[40, conn] - 0.5*phi_k[2]*(delx * gpuGlobalDataCommon[44, conn] + dely * gpuGlobalDataCommon[48, conn]),
                            gpuGlobalDataCommon[41, conn] - 0.5*phi_k[3]*(delx * gpuGlobalDataCommon[45, conn] + dely * gpuGlobalDataCommon[49, conn]),
                            gpuGlobalDataCommon[42, conn] - 0.5*phi_k[4]*(delx * gpuGlobalDataCommon[46, conn] + dely * gpuGlobalDataCommon[50, conn])
                        )
        end

        if limiter_flag == 2
            @cuprintf("\n Havent written the code - die \n")
        end

        qtilde_to_primitive_kernel(qtilde_i, gpuConfigData, result)
        flux_Gxn_kernel(nx, ny, result[1], result[2], result[3], result[4], G_i)
        qtilde_to_primitive_kernel(qtilde_k, gpuConfigData, result)
        flux_Gxn_kernel(nx, ny, result[1], result[2], result[3], result[4], G_k)
        # CUDAnative.synchronize()
        for i in 1:4
            sum_delx_delf[i] += (G_k[i] - G_i[i]) * dels_weights
            sum_dely_delf[i] += (G_k[i] - G_i[i]) * deln_weights
        end
    end

    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1.0 / det
    for i in 1:4
        Gxn[i] = (sum_delx_delf[i]*sum_dely_sqr - sum_dely_delf[i]*sum_delx_dely)*one_by_det
    end
    return nothing
end

function interior_dGy_pos_kernel(gpuGlobalDataCommon, idx, gpuConfigData, Gyp, phi_i, phi_k, G_i, G_k, sum_delx_delf, sum_dely_delf, result)

    power = gpuConfigData[6]
    limiter_flag = gpuConfigData[7]

    fill!(phi_i, 0.0)
    fill!(phi_i, 0.0)
    fill!(G_i, 0.0)
    fill!(G_k, 0.0)

    sum_delx_sqr = 0.0
    sum_dely_sqr = 0.0
    sum_delx_dely = 0.0

    fill!(sum_delx_delf,0.0)
    fill!(sum_dely_delf,0.0)
    qtilde_i = (0,0,0,0)
    qtilde_k = (0,0,0,0)

    x_i = gpuGlobalDataCommon[2, idx]
    y_i = gpuGlobalDataCommon[3, idx]
    nx = gpuGlobalDataCommon[29, idx]
    ny = gpuGlobalDataCommon[30, idx]

    tx = ny
    ty = -nx

    for iter in 96:115
        conn = Int(gpuGlobalDataCommon[iter, idx])
        if conn == 0.0
            break
        end

        x_k = gpuGlobalDataCommon[2, conn]
        y_k = gpuGlobalDataCommon[3, conn]
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

        qtilde_i =  (
                        gpuGlobalDataCommon[39, idx] - 0.5*(delx * gpuGlobalDataCommon[43, idx] + dely * gpuGlobalDataCommon[47, idx]),
                        gpuGlobalDataCommon[40, idx] - 0.5*(delx * gpuGlobalDataCommon[44, idx] + dely * gpuGlobalDataCommon[48, idx]),
                        gpuGlobalDataCommon[41, idx] - 0.5*(delx * gpuGlobalDataCommon[45, idx] + dely * gpuGlobalDataCommon[49, idx]),
                        gpuGlobalDataCommon[42, idx] - 0.5*(delx * gpuGlobalDataCommon[46, idx] + dely * gpuGlobalDataCommon[50, idx])
                    )
        qtilde_k = (
                        gpuGlobalDataCommon[39, conn] - 0.5*(delx * gpuGlobalDataCommon[43, conn] + dely * gpuGlobalDataCommon[47, conn]),
                        gpuGlobalDataCommon[40, conn] - 0.5*(delx * gpuGlobalDataCommon[44, conn] + dely * gpuGlobalDataCommon[48, conn]),
                        gpuGlobalDataCommon[41, conn] - 0.5*(delx * gpuGlobalDataCommon[45, conn] + dely * gpuGlobalDataCommon[49, conn]),
                        gpuGlobalDataCommon[42, conn] - 0.5*(delx * gpuGlobalDataCommon[46, conn] + dely * gpuGlobalDataCommon[50, conn])
                    )

        if limiter_flag == 1
            venkat_limiter_kernel(qtilde_i, gpuGlobalDataCommon, idx, gpuConfigData, phi_i)
            venkat_limiter_kernel(qtilde_k, gpuGlobalDataCommon, conn, gpuConfigData, phi_k)
            # CUDAnative.synchronize()
            qtilde_i =  (
                            gpuGlobalDataCommon[39, idx] - 0.5*phi_i[1]*(delx * gpuGlobalDataCommon[43, idx] + dely * gpuGlobalDataCommon[47, idx]),
                            gpuGlobalDataCommon[40, idx] - 0.5*phi_i[2]*(delx * gpuGlobalDataCommon[44, idx] + dely * gpuGlobalDataCommon[48, idx]),
                            gpuGlobalDataCommon[41, idx] - 0.5*phi_i[3]*(delx * gpuGlobalDataCommon[45, idx] + dely * gpuGlobalDataCommon[49, idx]),
                            gpuGlobalDataCommon[42, idx] - 0.5*phi_i[4]*(delx * gpuGlobalDataCommon[46, idx] + dely * gpuGlobalDataCommon[50, idx])
                        )
            qtilde_k = (
                            gpuGlobalDataCommon[39, conn] - 0.5*phi_k[1]*(delx * gpuGlobalDataCommon[43, conn] + dely * gpuGlobalDataCommon[47, conn]),
                            gpuGlobalDataCommon[40, conn] - 0.5*phi_k[2]*(delx * gpuGlobalDataCommon[44, conn] + dely * gpuGlobalDataCommon[48, conn]),
                            gpuGlobalDataCommon[41, conn] - 0.5*phi_k[3]*(delx * gpuGlobalDataCommon[45, conn] + dely * gpuGlobalDataCommon[49, conn]),
                            gpuGlobalDataCommon[42, conn] - 0.5*phi_k[4]*(delx * gpuGlobalDataCommon[46, conn] + dely * gpuGlobalDataCommon[50, conn])
                        )
        end

        if limiter_flag == 2
            @cuprintf("\n Havent written the code - die \n")
        end

        qtilde_to_primitive_kernel(qtilde_i, gpuConfigData, result)
        flux_Gyp_kernel(nx, ny, result[1], result[2], result[3], result[4], G_i)
        qtilde_to_primitive_kernel(qtilde_k, gpuConfigData, result)
        flux_Gyp_kernel(nx, ny, result[1], result[2], result[3], result[4], G_k)
        # CUDAnative.synchronize()
        for i in 1:4
            sum_delx_delf[i] += (G_k[i] - G_i[i]) * dels_weights
            sum_dely_delf[i] += (G_k[i] - G_i[i]) * deln_weights
        end
    end

    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1.0 / det
    for i in 1:4
        Gyp[i] = (sum_delx_delf[i]*sum_dely_sqr - sum_dely_delf[i]*sum_delx_dely)*one_by_det
    end
    return nothing
end

function interior_dGy_neg_kernel(gpuGlobalDataCommon, idx, gpuConfigData, Gyn, phi_i, phi_k, G_i, G_k, sum_delx_delf, sum_dely_delf, result)

    power = gpuConfigData[6]
    limiter_flag = gpuConfigData[7]

    fill!(phi_i, 0.0)
    fill!(phi_i, 0.0)
    fill!(G_i, 0.0)
    fill!(G_k, 0.0)

    sum_delx_sqr = 0.0
    sum_dely_sqr = 0.0
    sum_delx_dely = 0.0

    fill!(sum_delx_delf,0.0)
    fill!(sum_dely_delf,0.0)
    qtilde_i = (0,0,0,0)
    qtilde_k = (0,0,0,0)

    x_i = gpuGlobalDataCommon[2, idx]
    y_i = gpuGlobalDataCommon[3, idx]
    nx = gpuGlobalDataCommon[29, idx]
    ny = gpuGlobalDataCommon[30, idx]

    tx = ny
    ty = -nx

    for iter in 116:135
        conn = Int(gpuGlobalDataCommon[iter, idx])
        if conn == 0.0
            break
        end

        x_k = gpuGlobalDataCommon[2, conn]
        y_k = gpuGlobalDataCommon[3, conn]
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

        qtilde_i =  (
                        gpuGlobalDataCommon[39, idx] - 0.5*(delx * gpuGlobalDataCommon[43, idx] + dely * gpuGlobalDataCommon[47, idx]),
                        gpuGlobalDataCommon[40, idx] - 0.5*(delx * gpuGlobalDataCommon[44, idx] + dely * gpuGlobalDataCommon[48, idx]),
                        gpuGlobalDataCommon[41, idx] - 0.5*(delx * gpuGlobalDataCommon[45, idx] + dely * gpuGlobalDataCommon[49, idx]),
                        gpuGlobalDataCommon[42, idx] - 0.5*(delx * gpuGlobalDataCommon[46, idx] + dely * gpuGlobalDataCommon[50, idx])
                    )
        qtilde_k = (
                        gpuGlobalDataCommon[39, conn] - 0.5*(delx * gpuGlobalDataCommon[43, conn] + dely * gpuGlobalDataCommon[47, conn]),
                        gpuGlobalDataCommon[40, conn] - 0.5*(delx * gpuGlobalDataCommon[44, conn] + dely * gpuGlobalDataCommon[48, conn]),
                        gpuGlobalDataCommon[41, conn] - 0.5*(delx * gpuGlobalDataCommon[45, conn] + dely * gpuGlobalDataCommon[49, conn]),
                        gpuGlobalDataCommon[42, conn] - 0.5*(delx * gpuGlobalDataCommon[46, conn] + dely * gpuGlobalDataCommon[50, conn])
                    )

        if limiter_flag == 1
            venkat_limiter_kernel(qtilde_i, gpuGlobalDataCommon, idx, gpuConfigData, phi_i)
            venkat_limiter_kernel(qtilde_k, gpuGlobalDataCommon, conn, gpuConfigData, phi_k)
            # CUDAnative.synchronize()
            qtilde_i =  (
                            gpuGlobalDataCommon[39, idx] - 0.5*phi_i[1]*(delx * gpuGlobalDataCommon[43, idx] + dely * gpuGlobalDataCommon[47, idx]),
                            gpuGlobalDataCommon[40, idx] - 0.5*phi_i[2]*(delx * gpuGlobalDataCommon[44, idx] + dely * gpuGlobalDataCommon[48, idx]),
                            gpuGlobalDataCommon[41, idx] - 0.5*phi_i[3]*(delx * gpuGlobalDataCommon[45, idx] + dely * gpuGlobalDataCommon[49, idx]),
                            gpuGlobalDataCommon[42, idx] - 0.5*phi_i[4]*(delx * gpuGlobalDataCommon[46, idx] + dely * gpuGlobalDataCommon[50, idx])
                        )
            qtilde_k = (
                            gpuGlobalDataCommon[39, conn] - 0.5*phi_k[1]*(delx * gpuGlobalDataCommon[43, conn] + dely * gpuGlobalDataCommon[47, conn]),
                            gpuGlobalDataCommon[40, conn] - 0.5*phi_k[2]*(delx * gpuGlobalDataCommon[44, conn] + dely * gpuGlobalDataCommon[48, conn]),
                            gpuGlobalDataCommon[41, conn] - 0.5*phi_k[3]*(delx * gpuGlobalDataCommon[45, conn] + dely * gpuGlobalDataCommon[49, conn]),
                            gpuGlobalDataCommon[42, conn] - 0.5*phi_k[4]*(delx * gpuGlobalDataCommon[46, conn] + dely * gpuGlobalDataCommon[50, conn])
                        )
        end

        if limiter_flag == 2
            @cuprintf("\n Havent written the code - die \n")
        end

        qtilde_to_primitive_kernel(qtilde_i, gpuConfigData, result)
        flux_Gyn_kernel(nx, ny, result[1], result[2], result[3], result[4], G_i)
        qtilde_to_primitive_kernel(qtilde_k, gpuConfigData, result)
        flux_Gyn_kernel(nx, ny, result[1], result[2], result[3], result[4], G_k)
        # CUDAnative.synchronize()
        for i in 1:4
            sum_delx_delf[i] += (G_k[i] - G_i[i]) * dels_weights
            sum_dely_delf[i] += (G_k[i] - G_i[i]) * deln_weights
        end
    end

    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1.0 / det
    for i in 1:4
        Gyn[i] = (sum_delx_delf[i]*sum_dely_sqr - sum_dely_delf[i]*sum_delx_dely)*one_by_det
    end
    return nothing
end
