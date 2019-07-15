
function interior_dGx_pos_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, idx, gpuConfigData, power, limiter_flag, gamma, shared)

    sum_delx_sqr = 0.0
    sum_dely_sqr = 0.0
    sum_delx_dely = 0.0
    sum_1, sum_2, sum_3, sum_4 = 0.0,0.0,0.0,0.0
    sum_5, sum_6, sum_7, sum_8 = 0.0,0.0,0.0,0.0

    x_i = gpuGlobalDataFixedPoint[idx].x
    y_i = gpuGlobalDataFixedPoint[idx].y
    nx = gpuGlobalDataFixedPoint[idx].nx
    ny = gpuGlobalDataFixedPoint[idx].ny

    tx = ny
    ty = -nx

    for iter in 15:24
        conn = gpuGlobalDataConn[iter, idx]
        if conn == 0
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

        # if limiter_flag == 1
            venkat_limiter_kernel_i(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, idx, gpuConfigData, delx, dely)
            venkat_limiter_kernel_k(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, conn, gpuConfigData, idx, delx, dely)
            # CUDAnative.synchronize()
            qtilde_i1, qtilde_i2, qtilde_i3, qtilde_i4 =
                            gpuGlobalDataRest[9, idx] - 0.5*gpuGlobalDataRest[29, idx]*(delx * gpuGlobalDataRest[13, idx] + dely * gpuGlobalDataRest[17, idx]),
                            gpuGlobalDataRest[10, idx] - 0.5*gpuGlobalDataRest[30, idx]*(delx * gpuGlobalDataRest[14, idx] + dely * gpuGlobalDataRest[18, idx]),
                            gpuGlobalDataRest[11, idx] - 0.5*gpuGlobalDataRest[31, idx]*(delx * gpuGlobalDataRest[15, idx] + dely * gpuGlobalDataRest[19, idx]),
                            gpuGlobalDataRest[12, idx] - 0.5*gpuGlobalDataRest[32, idx]*(delx * gpuGlobalDataRest[16, idx] + dely * gpuGlobalDataRest[20, idx])
            qtilde_k1, qtilde_k2, qtilde_k3, qtilde_k4 =
                            gpuGlobalDataRest[9, conn] - 0.5*gpuGlobalDataRest[33, idx]*(delx * gpuGlobalDataRest[13, conn] + dely * gpuGlobalDataRest[17, conn]),
                            gpuGlobalDataRest[10, conn] - 0.5*gpuGlobalDataRest[34, idx]*(delx * gpuGlobalDataRest[14, conn] + dely * gpuGlobalDataRest[18, conn]),
                            gpuGlobalDataRest[11, conn] - 0.5*gpuGlobalDataRest[35, idx]*(delx * gpuGlobalDataRest[15, conn] + dely * gpuGlobalDataRest[19, conn]),
                            gpuGlobalDataRest[12, conn] - 0.5*gpuGlobalDataRest[36, idx]*(delx * gpuGlobalDataRest[16, conn] + dely * gpuGlobalDataRest[20, conn])
        # end

        # if limiter_flag == 2
        #     @cuprintf("\n Havent written the code - die \n")
        # end
        qtilde_to_primitive_kernel(qtilde_i1, qtilde_i2, qtilde_i3, qtilde_i4, gamma, gpuGlobalDataRest, shared, idx)
        flux_Gxp_kernel(nx, ny, gpuGlobalDataRest, idx, shared, 1)
        qtilde_to_primitive_kernel(qtilde_k1, qtilde_k2, qtilde_k3, qtilde_k4, gamma, gpuGlobalDataRest, shared, idx)
        flux_Gxp_kernel(nx, ny, gpuGlobalDataRest, idx, shared, 2)
        # CUDAnative.synchronize()
        temp = gpuGlobalDataRest[41, idx] - gpuGlobalDataRest[37, idx]
        sum_1 += (temp) * dels_weights
        sum_5 += (temp) * deln_weights
        temp = gpuGlobalDataRest[42, idx] - gpuGlobalDataRest[38, idx]
        sum_2 += (temp) * dels_weights
        sum_6 += (temp) * deln_weights
        temp = gpuGlobalDataRest[43, idx] - gpuGlobalDataRest[39, idx]
        sum_3 += (temp) * dels_weights
        sum_7 += (temp) * deln_weights
        temp = gpuGlobalDataRest[44, idx] - gpuGlobalDataRest[40, idx]
        sum_4 += (temp) * dels_weights
        sum_8 += (temp) * deln_weights
    end

    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1.0 / det

    gpuGlobalDataRest[5, idx] += (sum_1*sum_dely_sqr - sum_5*sum_delx_dely)*one_by_det
    gpuGlobalDataRest[6, idx] += (sum_2*sum_dely_sqr - sum_6*sum_delx_dely)*one_by_det
    gpuGlobalDataRest[7, idx] += (sum_3*sum_dely_sqr - sum_7*sum_delx_dely)*one_by_det
    gpuGlobalDataRest[8, idx] += (sum_4*sum_dely_sqr - sum_8*sum_delx_dely)*one_by_det
    # if idx == 1
    #     @cuprintf("\n %f %f %f %f", gpuGlobalDataRest[5, idx],gpuGlobalDataRest[6, idx],gpuGlobalDataRest[7, idx],gpuGlobalDataRest[8, idx])
    # end
    return nothing
end

function interior_dGx_neg_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, idx, gpuConfigData, power, limiter_flag, gamma, shared)

    sum_delx_sqr = 0.0
    sum_dely_sqr = 0.0
    sum_delx_dely = 0.0
    sum_1, sum_2, sum_3, sum_4 = 0.0,0.0,0.0,0.0
    sum_5, sum_6, sum_7, sum_8 = 0.0,0.0,0.0,0.0


    x_i = gpuGlobalDataFixedPoint[idx].x
    y_i = gpuGlobalDataFixedPoint[idx].y
    nx = gpuGlobalDataFixedPoint[idx].nx
    ny = gpuGlobalDataFixedPoint[idx].ny

    tx = ny
    ty = -nx

    for iter in 25:34
        conn = gpuGlobalDataConn[iter, idx]
        if conn == 0
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

        # if limiter_flag == 1
            venkat_limiter_kernel_i(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, idx, gpuConfigData, delx, dely)
            venkat_limiter_kernel_k(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, conn, gpuConfigData, idx, delx, dely)
            # CUDAnative.synchronize()
            qtilde_i1, qtilde_i2, qtilde_i3, qtilde_i4=
                            gpuGlobalDataRest[9, idx] - 0.5*gpuGlobalDataRest[29, idx]*(delx * gpuGlobalDataRest[13, idx] + dely * gpuGlobalDataRest[17, idx]),
                            gpuGlobalDataRest[10, idx] - 0.5*gpuGlobalDataRest[30, idx]*(delx * gpuGlobalDataRest[14, idx] + dely * gpuGlobalDataRest[18, idx]),
                            gpuGlobalDataRest[11, idx] - 0.5*gpuGlobalDataRest[31, idx]*(delx * gpuGlobalDataRest[15, idx] + dely * gpuGlobalDataRest[19, idx]),
                            gpuGlobalDataRest[12, idx] - 0.5*gpuGlobalDataRest[32, idx]*(delx * gpuGlobalDataRest[16, idx] + dely * gpuGlobalDataRest[20, idx])
            qtilde_k1, qtilde_k2, qtilde_k3, qtilde_k4 =
                            gpuGlobalDataRest[9, conn] - 0.5*gpuGlobalDataRest[33, idx]*(delx * gpuGlobalDataRest[13, conn] + dely * gpuGlobalDataRest[17, conn]),
                            gpuGlobalDataRest[10, conn] - 0.5*gpuGlobalDataRest[34, idx]*(delx * gpuGlobalDataRest[14, conn] + dely * gpuGlobalDataRest[18, conn]),
                            gpuGlobalDataRest[11, conn] - 0.5*gpuGlobalDataRest[35, idx]*(delx * gpuGlobalDataRest[15, conn] + dely * gpuGlobalDataRest[19, conn]),
                            gpuGlobalDataRest[12, conn] - 0.5*gpuGlobalDataRest[36, idx]*(delx * gpuGlobalDataRest[16, conn] + dely * gpuGlobalDataRest[20, conn])
        # end

        # if limiter_flag == 2
        #     @cuprintf("\n Havent written the code - die \n")
        # end
        qtilde_to_primitive_kernel(qtilde_i1, qtilde_i2, qtilde_i3, qtilde_i4, gamma, gpuGlobalDataRest, shared, idx)
        flux_Gxn_kernel(nx, ny, gpuGlobalDataRest, idx, shared, 1)
        qtilde_to_primitive_kernel(qtilde_k1, qtilde_k2, qtilde_k3, qtilde_k4, gamma, gpuGlobalDataRest, shared, idx)
        flux_Gxn_kernel(nx, ny, gpuGlobalDataRest, idx, shared, 2)
        # CUDAnative.synchronize()
        temp = gpuGlobalDataRest[41, idx] - gpuGlobalDataRest[37, idx]
        sum_1 += (temp) * dels_weights
        sum_5 += (temp) * deln_weights
        temp = gpuGlobalDataRest[42, idx] - gpuGlobalDataRest[38, idx]
        sum_2 += (temp) * dels_weights
        sum_6 += (temp) * deln_weights
        temp = gpuGlobalDataRest[43, idx] - gpuGlobalDataRest[39, idx]
        sum_3 += (temp) * dels_weights
        sum_7 += (temp) * deln_weights
        temp = gpuGlobalDataRest[44, idx] - gpuGlobalDataRest[40, idx]
        sum_4 += (temp) * dels_weights
        sum_8 += (temp) * deln_weights
    end

    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1.0 / det
    # if idx == 1
    #     @cuprintf("\n===Gyn===")
    #     @cuprintf("\n %f", sum_delx_sqr)
    #     @cuprintf("\n %f", sum_dely_sqr)
    #     @cuprintf("\n %f", sum_delx_dely)
    #     @cuprintf("\n %f", det)
    #     @cuprintf("\n %.17f %.17f %.17f %.17f", sum_1,sum_2,sum_3,sum_4)
    #     @cuprintf("\n %.17f %.17f %.17f %.17f", sum_5,sum_6,sum_7,sum_8)
    # end
    gpuGlobalDataRest[5, idx] += (sum_1*sum_dely_sqr - sum_5*sum_delx_dely)*one_by_det
    gpuGlobalDataRest[6, idx] += (sum_2*sum_dely_sqr - sum_6*sum_delx_dely)*one_by_det
    gpuGlobalDataRest[7, idx] += (sum_3*sum_dely_sqr - sum_7*sum_delx_dely)*one_by_det
    gpuGlobalDataRest[8, idx] += (sum_4*sum_dely_sqr - sum_8*sum_delx_dely)*one_by_det
    # if idx == 1
    #     @cuprintf("\n %f %f %f %f", gpuGlobalDataRest[5, idx],gpuGlobalDataRest[6, idx],gpuGlobalDataRest[7, idx],gpuGlobalDataRest[8, idx])
    # end
    return nothing
end

function interior_dGy_pos_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, idx, gpuConfigData, power, limiter_flag, gamma, shared)

    sum_delx_sqr = 0.0
    sum_dely_sqr = 0.0
    sum_delx_dely = 0.0
    sum_1, sum_2, sum_3, sum_4 = 0.0,0.0,0.0,0.0
    sum_5, sum_6, sum_7, sum_8 = 0.0,0.0,0.0,0.0

    x_i = gpuGlobalDataFixedPoint[idx].x
    y_i = gpuGlobalDataFixedPoint[idx].y
    nx = gpuGlobalDataFixedPoint[idx].nx
    ny = gpuGlobalDataFixedPoint[idx].ny

    tx = ny
    ty = -nx

    for iter in 35:44
        conn = gpuGlobalDataConn[iter, idx]
        if conn == 0
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

        # if limiter_flag == 1
            venkat_limiter_kernel_i(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, idx, gpuConfigData, delx, dely)
            venkat_limiter_kernel_k(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, conn, gpuConfigData, idx, delx, dely)
            # CUDAnative.synchronize()
            qtilde_i1, qtilde_i2, qtilde_i3, qtilde_i4=
                            gpuGlobalDataRest[9, idx] - 0.5*gpuGlobalDataRest[29, idx]*(delx * gpuGlobalDataRest[13, idx] + dely * gpuGlobalDataRest[17, idx]),
                            gpuGlobalDataRest[10, idx] - 0.5*gpuGlobalDataRest[30, idx]*(delx * gpuGlobalDataRest[14, idx] + dely * gpuGlobalDataRest[18, idx]),
                            gpuGlobalDataRest[11, idx] - 0.5*gpuGlobalDataRest[31, idx]*(delx * gpuGlobalDataRest[15, idx] + dely * gpuGlobalDataRest[19, idx]),
                            gpuGlobalDataRest[12, idx] - 0.5*gpuGlobalDataRest[32, idx]*(delx * gpuGlobalDataRest[16, idx] + dely * gpuGlobalDataRest[20, idx])
            qtilde_k1, qtilde_k2, qtilde_k3, qtilde_k4 =
                            gpuGlobalDataRest[9, conn] - 0.5*gpuGlobalDataRest[33, idx]*(delx * gpuGlobalDataRest[13, conn] + dely * gpuGlobalDataRest[17, conn]),
                            gpuGlobalDataRest[10, conn] - 0.5*gpuGlobalDataRest[34, idx]*(delx * gpuGlobalDataRest[14, conn] + dely * gpuGlobalDataRest[18, conn]),
                            gpuGlobalDataRest[11, conn] - 0.5*gpuGlobalDataRest[35, idx]*(delx * gpuGlobalDataRest[15, conn] + dely * gpuGlobalDataRest[19, conn]),
                            gpuGlobalDataRest[12, conn] - 0.5*gpuGlobalDataRest[36, idx]*(delx * gpuGlobalDataRest[16, conn] + dely * gpuGlobalDataRest[20, conn])
        # end

        # if limiter_flag == 2
        #     @cuprintf("\n Havent written the code - die \n")
        # end
        qtilde_to_primitive_kernel(qtilde_i1, qtilde_i2, qtilde_i3, qtilde_i4, gamma, gpuGlobalDataRest, shared, idx)
        flux_Gyp_kernel(nx, ny, gpuGlobalDataRest, idx, shared, 1)
        qtilde_to_primitive_kernel(qtilde_k1, qtilde_k2, qtilde_k3, qtilde_k4, gamma, gpuGlobalDataRest, shared, idx)
        flux_Gyp_kernel(nx, ny, gpuGlobalDataRest, idx, shared, 2)
        # CUDAnative.synchronize()
        temp = gpuGlobalDataRest[41, idx] - gpuGlobalDataRest[37, idx]
        sum_1 += (temp) * dels_weights
        sum_5 += (temp) * deln_weights
        temp = gpuGlobalDataRest[42, idx] - gpuGlobalDataRest[38, idx]
        sum_2 += (temp) * dels_weights
        sum_6 += (temp) * deln_weights
        temp = gpuGlobalDataRest[43, idx] - gpuGlobalDataRest[39, idx]
        sum_3 += (temp) * dels_weights
        sum_7 += (temp) * deln_weights
        temp = gpuGlobalDataRest[44, idx] - gpuGlobalDataRest[40, idx]
        sum_4 += (temp) * dels_weights
        sum_8 += (temp) * deln_weights
        # if idx == 200
        #     @cuprintf("\n %d", conn)
        #     @cuprintf("\n %.17f %.17f %.17f %.17f", gpuGlobalDataRest[45, idx], gpuGlobalDataRest[46, idx], gpuGlobalDataRest[47, idx], gpuGlobalDataRest[48, idx])
        #     @cuprintf("\n %.17f %.17f %.17f %.17f", gpuGlobalDataRest[37, idx], gpuGlobalDataRest[38, idx], gpuGlobalDataRest[39, idx], gpuGlobalDataRest[40, idx])
        #     @cuprintf("\n %.17f %.17f %.17f %.17f", gpuGlobalDataRest[41, idx], gpuGlobalDataRest[42, idx], gpuGlobalDataRest[43, idx], gpuGlobalDataRest[44, idx])
        # end
    end

    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1.0 / det
    gpuGlobalDataRest[5, idx] += (sum_5*sum_delx_sqr - sum_1*sum_delx_dely)*one_by_det
    gpuGlobalDataRest[6, idx] += (sum_6*sum_delx_sqr - sum_2*sum_delx_dely)*one_by_det
    gpuGlobalDataRest[7, idx] += (sum_7*sum_delx_sqr - sum_3*sum_delx_dely)*one_by_det
    gpuGlobalDataRest[8, idx] += (sum_8*sum_delx_sqr - sum_4*sum_delx_dely)*one_by_det
    # if idx ==1
    #     @cuprintf("\n %f %f %f %f", gpuGlobalDataRest[5, idx],gpuGlobalDataRest[6, idx],gpuGlobalDataRest[7, idx],gpuGlobalDataRest[8, idx])
    # end
    return nothing
end

function interior_dGy_neg_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, idx, gpuConfigData, power, limiter_flag, gamma, shared)

    sum_delx_sqr = 0.0
    sum_dely_sqr = 0.0
    sum_delx_dely = 0.0
    sum_1, sum_2, sum_3, sum_4 = 0.0,0.0,0.0,0.0
    sum_5, sum_6, sum_7, sum_8 = 0.0,0.0,0.0,0.0

    x_i = gpuGlobalDataFixedPoint[idx].x
    y_i = gpuGlobalDataFixedPoint[idx].y
    nx = gpuGlobalDataFixedPoint[idx].nx
    ny = gpuGlobalDataFixedPoint[idx].ny

    tx = ny
    ty = -nx

    for iter in 45:54
        conn = gpuGlobalDataConn[iter, idx]
        if conn == 0
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

        # if limiter_flag == 1
            venkat_limiter_kernel_i(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, idx, gpuConfigData, delx, dely)
            venkat_limiter_kernel_k(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, conn, gpuConfigData, idx, delx, dely)
            # CUDAnative.synchronize()
            qtilde_i1, qtilde_i2, qtilde_i3, qtilde_i4=
                            gpuGlobalDataRest[9, idx] - 0.5*gpuGlobalDataRest[29, idx]*(delx * gpuGlobalDataRest[13, idx] + dely * gpuGlobalDataRest[17, idx]),
                            gpuGlobalDataRest[10, idx] - 0.5*gpuGlobalDataRest[30, idx]*(delx * gpuGlobalDataRest[14, idx] + dely * gpuGlobalDataRest[18, idx]),
                            gpuGlobalDataRest[11, idx] - 0.5*gpuGlobalDataRest[31, idx]*(delx * gpuGlobalDataRest[15, idx] + dely * gpuGlobalDataRest[19, idx]),
                            gpuGlobalDataRest[12, idx] - 0.5*gpuGlobalDataRest[32, idx]*(delx * gpuGlobalDataRest[16, idx] + dely * gpuGlobalDataRest[20, idx])
            qtilde_k1, qtilde_k2, qtilde_k3, qtilde_k4 =
                            gpuGlobalDataRest[9, conn] - 0.5*gpuGlobalDataRest[33, idx]*(delx * gpuGlobalDataRest[13, conn] + dely * gpuGlobalDataRest[17, conn]),
                            gpuGlobalDataRest[10, conn] - 0.5*gpuGlobalDataRest[34, idx]*(delx * gpuGlobalDataRest[14, conn] + dely * gpuGlobalDataRest[18, conn]),
                            gpuGlobalDataRest[11, conn] - 0.5*gpuGlobalDataRest[35, idx]*(delx * gpuGlobalDataRest[15, conn] + dely * gpuGlobalDataRest[19, conn]),
                            gpuGlobalDataRest[12, conn] - 0.5*gpuGlobalDataRest[36, idx]*(delx * gpuGlobalDataRest[16, conn] + dely * gpuGlobalDataRest[20, conn])
        # end
        # if limiter_flag == 2
        #     @cuprintf("\n Havent written the code - die \n")
        # end
        qtilde_to_primitive_kernel(qtilde_i1, qtilde_i2, qtilde_i3, qtilde_i4, gamma, gpuGlobalDataRest, shared, idx)
        flux_Gyn_kernel(nx, ny, gpuGlobalDataRest, idx, shared, 1)
        qtilde_to_primitive_kernel(qtilde_k1, qtilde_k2, qtilde_k3, qtilde_k4, gamma, gpuGlobalDataRest, shared, idx)
        flux_Gyn_kernel(nx, ny, gpuGlobalDataRest, idx, shared, 2)
        # CUDAnative.synchronize()
        temp = gpuGlobalDataRest[41, idx] - gpuGlobalDataRest[37, idx]
        sum_1 += (temp) * dels_weights
        sum_5 += (temp) * deln_weights
        temp = gpuGlobalDataRest[42, idx] - gpuGlobalDataRest[38, idx]
        sum_2 += (temp) * dels_weights
        sum_6 += (temp) * deln_weights
        temp = gpuGlobalDataRest[43, idx] - gpuGlobalDataRest[39, idx]
        sum_3 += (temp) * dels_weights
        sum_7 += (temp) * deln_weights
        temp = gpuGlobalDataRest[44, idx] - gpuGlobalDataRest[40, idx]
        sum_4 += (temp) * dels_weights
        sum_8 += (temp) * deln_weights
        # if idx == 1
        #     @cuprintf("\n %d", conn)
        #     @cuprintf("\n %.17f %.17f %.17f %.17f", gpuGlobalDataRest[45, idx], gpuGlobalDataRest[46, idx], gpuGlobalDataRest[47, idx], gpuGlobalDataRest[48, idx])
        #     @cuprintf("\n %.17f %.17f %.17f %.17f", gpuGlobalDataRest[37, idx], gpuGlobalDataRest[38, idx], gpuGlobalDataRest[39, idx], gpuGlobalDataRest[40, idx])
        #     @cuprintf("\n %.17f %.17f %.17f %.17f", gpuGlobalDataRest[41, idx], gpuGlobalDataRest[42, idx], gpuGlobalDataRest[43, idx], gpuGlobalDataRest[44, idx])
        # end
    end

    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1.0 / det
    gpuGlobalDataRest[5, idx] += (sum_5*sum_delx_sqr - sum_1*sum_delx_dely)*one_by_det
    gpuGlobalDataRest[6, idx] += (sum_6*sum_delx_sqr - sum_2*sum_delx_dely)*one_by_det
    gpuGlobalDataRest[7, idx] += (sum_7*sum_delx_sqr - sum_3*sum_delx_dely)*one_by_det
    gpuGlobalDataRest[8, idx] += (sum_8*sum_delx_sqr - sum_4*sum_delx_dely)*one_by_det
    # if idx ==1
    #     @cuprintf("\n %f %f %f %f", gpuGlobalDataRest[5, idx],gpuGlobalDataRest[6, idx],gpuGlobalDataRest[7, idx],gpuGlobalDataRest[8, idx])
    # end
    return nothing
end
