function wall_dGx_pos_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, idx, gpuConfigData)
    power = gpuConfigData[6]
    limiter_flag = gpuConfigData[7]
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
    sum_1, sum_2,sum_3,sum_4 = 0.0,0.0,0.0,0.0
    sum_5, sum_6,sum_7,sum_8 = 0.0,0.0,0.0,0.0



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

        # if idx == 3
        #     @cuprintf("\n %d", conn)
        #     @cuprintf("\n %.17f %.17f %.17f %.17f", qtilde_k[1], qtilde_k[2], qtilde_k[3], qtilde_k[4])
        # end


        if limiter_flag == 1
            venkat_limiter_kernel_i(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, idx, gpuConfigData, delx, dely)
            venkat_limiter_kernel_k(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, conn, gpuConfigData, idx, delx, dely)
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
        # if idx == 3
        #     @cuprintf("\n %d", conn)
        #     @cuprintf("\n %.17f %.17f %.17f %.17f", gpuGlobalDataRest[29, idx], gpuGlobalDataRest[30, idx], gpuGlobalDataRest[31, idx], gpuGlobalDataRest[32, idx])
        #     @cuprintf("\n %.17f %.17f %.17f %.17f", gpuGlobalDataRest[33,conn], gpuGlobalDataRest[34,conn], gpuGlobalDataRest[35,conn], gpuGlobalDataRest[36,conn])
        #     @cuprintf("\n %.17f %.17f %.17f %.17f", qtilde_k[1], qtilde_k[2], qtilde_k[3], qtilde_k[4])
        # end
        if limiter_flag == 2
            @cuprintf("\n Havent written the code - die \n")
        end
        # if idx == 3
        #     @cuprintf("\n %d", conn)
        #     @cuprintf("\n %.17f %.17f %.17f %.17f", gpuGlobalDataRest[45, idx], gpuGlobalDataRest[46, idx], gpuGlobalDataRest[47, idx], gpuGlobalDataRest[48, idx])
        # end
        qtilde_to_primitive_kernel(qtilde_i, gpuConfigData, gpuGlobalDataRest, idx)
        # if idx == 3
        #     @cuprintf("\n %.17f %.17f %.17f %.17f", gpuGlobalDataRest[45, idx], gpuGlobalDataRest[46, idx], gpuGlobalDataRest[47, idx], gpuGlobalDataRest[48, idx])
        # end
        flux_quad_GxII_kernel(nx, ny, gpuGlobalDataRest, idx, 1)
        # if idx == 3
        #     @cuprintf("\n %.17f %.17f %.17f %.17f", gpuGlobalDataRest[45, idx], gpuGlobalDataRest[46, idx], gpuGlobalDataRest[47, idx], gpuGlobalDataRest[48, idx])
        # end
        qtilde_to_primitive_kernel(qtilde_k, gpuConfigData, gpuGlobalDataRest, idx)
        # if idx == 3
        #     @cuprintf("\n %.17f %.17f %.17f %.17f", gpuGlobalDataRest[45, idx], gpuGlobalDataRest[46, idx], gpuGlobalDataRest[47, idx], gpuGlobalDataRest[48, idx])
        # end
        flux_quad_GxII_kernel(nx, ny, gpuGlobalDataRest, idx, 2)
        # CUDAnative.synchronize()
        sum_1 += (gpuGlobalDataRest[41, idx] - gpuGlobalDataRest[37, idx]) * dels_weights
        sum_5 += (gpuGlobalDataRest[41, idx] - gpuGlobalDataRest[37, idx]) * deln_weights
        sum_2 += (gpuGlobalDataRest[42, idx] - gpuGlobalDataRest[38, idx]) * dels_weights
        sum_6 += (gpuGlobalDataRest[42, idx] - gpuGlobalDataRest[38, idx]) * deln_weights
        sum_3 += (gpuGlobalDataRest[43, idx] - gpuGlobalDataRest[39, idx]) * dels_weights
        sum_7 += (gpuGlobalDataRest[43, idx] - gpuGlobalDataRest[39, idx]) * deln_weights
        sum_4 += (gpuGlobalDataRest[44, idx] - gpuGlobalDataRest[40, idx]) * dels_weights
        sum_8 += (gpuGlobalDataRest[44, idx] - gpuGlobalDataRest[40, idx]) * deln_weights
        # if idx == 3
        #     @cuprintf("\n %d", conn)
        #     @cuprintf("\n %.17f %.17f %.17f %.17f", gpuGlobalDataRest[45, idx], gpuGlobalDataRest[46, idx], gpuGlobalDataRest[47, idx], gpuGlobalDataRest[48, idx])
        #     @cuprintf("\n %.17f %.17f %.17f %.17f", gpuGlobalDataRest[37, idx], gpuGlobalDataRest[38, idx], gpuGlobalDataRest[39, idx], gpuGlobalDataRest[40, idx])
        #     @cuprintf("\n %.17f %.17f %.17f %.17f", gpuGlobalDataRest[41, idx], gpuGlobalDataRest[42, idx], gpuGlobalDataRest[43, idx], gpuGlobalDataRest[44, idx])
        # end
    end

    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1.0 / det
    # if idx ==3
    #     @cuprintf("\n===Gn===")
    #     @cuprintf("\n %.17f %.17f %.17f %.17f", sum_1,sum_2,sum_3,sum_4)
    #     @cuprintf("\n %.17f %.17f %.17f %.17f", sum_5,sum_6,sum_7,sum_8)
    # end
    gpuGlobalDataRest[5, idx] += 2 * (sum_1*sum_dely_sqr - sum_5*sum_delx_dely)*one_by_det
    gpuGlobalDataRest[6, idx] += 2 * (sum_2*sum_dely_sqr - sum_6*sum_delx_dely)*one_by_det
    gpuGlobalDataRest[7, idx] += 2 * (sum_3*sum_dely_sqr - sum_7*sum_delx_dely)*one_by_det
    gpuGlobalDataRest[8, idx] += 2 * (sum_4*sum_dely_sqr - sum_8*sum_delx_dely)*one_by_det
    # if idx ==3
    #     @cuprintf("\n %.17f %.17f %.17f %.17f", gpuGlobalDataRest[5, idx],gpuGlobalDataRest[6, idx],gpuGlobalDataRest[7, idx],gpuGlobalDataRest[8, idx])
    # end
    return nothing
end

function wall_dGx_neg_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, idx, gpuConfigData)
    power = gpuConfigData[6]
    limiter_flag = gpuConfigData[7]

    # phi_i1,phi_i2,phi_i3,phi_i4 =  0.0,0.0,0.0,0.0
    # phi_k1,phi_k2,phi_k3,phi_k4 =  0.0,0.0,0.0,0.0
    # G_i1,G_i2,G_i3,G_i4 =  0.0,0.0,0.0,0.0
    # G_k1,G_k2,G_k3,G_k4 =  0.0,0.0,0.0,0.0

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
    sum_1, sum_2,sum_3,sum_4 = 0.0,0.0,0.0,0.0
    sum_5, sum_6,sum_7,sum_8 = 0.0,0.0,0.0,0.0

    # sum_delx_delf1,sum_delx_delf2,sum_delx_delf3,sum_delx_delf4 = 0.0,0.0,0.0,0.0
    # sum_dely_delf1,sum_dely_delf2,sum_dely_delf3,sum_dely_delf4 = 0.0,0.0,0.0,0.0




    # result1,result2,result3,result4 = 0.0,0.0,0.0,0.0

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

        if limiter_flag == 1
            venkat_limiter_kernel_i(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, idx, gpuConfigData, delx, dely)
            venkat_limiter_kernel_k(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, conn, gpuConfigData, idx, delx, dely)
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
        qtilde_to_primitive_kernel(qtilde_i, gpuConfigData, gpuGlobalDataRest, idx)
        flux_quad_GxI_kernel(nx, ny, gpuGlobalDataRest, idx, 1)
        qtilde_to_primitive_kernel(qtilde_k, gpuConfigData, gpuGlobalDataRest, idx)
        flux_quad_GxI_kernel(nx, ny, gpuGlobalDataRest, idx, 2)
        # CUDAnative.synchronize()
        sum_1 += (gpuGlobalDataRest[41, idx] - gpuGlobalDataRest[37, idx]) * dels_weights
        sum_5 += (gpuGlobalDataRest[41, idx] - gpuGlobalDataRest[37, idx]) * deln_weights
        sum_2 += (gpuGlobalDataRest[42, idx] - gpuGlobalDataRest[38, idx]) * dels_weights
        sum_6 += (gpuGlobalDataRest[42, idx] - gpuGlobalDataRest[38, idx]) * deln_weights
        sum_3 += (gpuGlobalDataRest[43, idx] - gpuGlobalDataRest[39, idx]) * dels_weights
        sum_7 += (gpuGlobalDataRest[43, idx] - gpuGlobalDataRest[39, idx]) * deln_weights
        sum_4 += (gpuGlobalDataRest[44, idx] - gpuGlobalDataRest[40, idx]) * dels_weights
        sum_8 += (gpuGlobalDataRest[44, idx] - gpuGlobalDataRest[40, idx]) * deln_weights
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

    gpuGlobalDataRest[5, idx] += 2 * (sum_1*sum_dely_sqr - sum_5*sum_delx_dely)*one_by_det
    gpuGlobalDataRest[6, idx] += 2 * (sum_2*sum_dely_sqr - sum_6*sum_delx_dely)*one_by_det
    gpuGlobalDataRest[7, idx] += 2 * (sum_3*sum_dely_sqr - sum_7*sum_delx_dely)*one_by_det
    gpuGlobalDataRest[8, idx] += 2 * (sum_4*sum_dely_sqr - sum_8*sum_delx_dely)*one_by_det
    # if idx ==3
    #     @cuprintf("\n %f %f %f %f", gpuGlobalDataRest[5, idx],gpuGlobalDataRest[6, idx],gpuGlobalDataRest[7, idx],gpuGlobalDataRest[8, idx])
    # end
    return nothing
end

function wall_dGy_neg_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, idx, gpuConfigData)

    power = gpuConfigData[6]
    limiter_flag = gpuConfigData[7]

    # phi_i1,phi_i2,phi_i3,phi_i4 =  0.0,0.0,0.0,0.0
    # phi_k1,phi_k2,phi_k3,phi_k4 =  0.0,0.0,0.0,0.0
    # G_i1,G_i2,G_i3,G_i4 =  0.0,0.0,0.0,0.0
    # G_k1,G_k2,G_k3,G_k4 =  0.0,0.0,0.0,0.0

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
    sum_1, sum_2,sum_3,sum_4 = 0.0,0.0,0.0,0.0
    sum_5, sum_6,sum_7,sum_8 = 0.0,0.0,0.0,0.0

    # sum_delx_delf1,sum_delx_delf2,sum_delx_delf3,sum_delx_delf4 = 0.0,0.0,0.0,0.0
    # sum_dely_delf1,sum_dely_delf2,sum_dely_delf3,sum_dely_delf4 = 0.0,0.0,0.0,0.0




    # result1,result2,result3,result4 = 0.0,0.0,0.0,0.0

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


        if limiter_flag == 1
            venkat_limiter_kernel_i(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, idx, gpuConfigData, delx, dely)
            venkat_limiter_kernel_k(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, conn, gpuConfigData, idx, delx, dely)
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
        qtilde_to_primitive_kernel(qtilde_i, gpuConfigData, gpuGlobalDataRest, idx)
        flux_Gyn_kernel(nx, ny, gpuGlobalDataRest, idx, 1)
        qtilde_to_primitive_kernel(qtilde_k, gpuConfigData, gpuGlobalDataRest, idx)
        flux_Gyn_kernel(nx, ny, gpuGlobalDataRest, idx, 2)
        # CUDAnative.synchronize()
        sum_1 += (gpuGlobalDataRest[41, idx] - gpuGlobalDataRest[37, idx]) * dels_weights
        sum_5 += (gpuGlobalDataRest[41, idx] - gpuGlobalDataRest[37, idx]) * deln_weights
        sum_2 += (gpuGlobalDataRest[42, idx] - gpuGlobalDataRest[38, idx]) * dels_weights
        sum_6 += (gpuGlobalDataRest[42, idx] - gpuGlobalDataRest[38, idx]) * deln_weights
        sum_3 += (gpuGlobalDataRest[43, idx] - gpuGlobalDataRest[39, idx]) * dels_weights
        sum_7 += (gpuGlobalDataRest[43, idx] - gpuGlobalDataRest[39, idx]) * deln_weights
        sum_4 += (gpuGlobalDataRest[44, idx] - gpuGlobalDataRest[40, idx]) * dels_weights
        sum_8 += (gpuGlobalDataRest[44, idx] - gpuGlobalDataRest[40, idx]) * deln_weights
        # if idx == 3
        #     @cuprintf("\n %d", conn)
        #     @cuprintf("\n %.17f %.17f %.17f %.17f", gpuGlobalDataRest[45, idx], gpuGlobalDataRest[46, idx], gpuGlobalDataRest[47, idx], gpuGlobalDataRest[48, idx])
        #     @cuprintf("\n %.17f %.17f %.17f %.17f", gpuGlobalDataRest[37, idx], gpuGlobalDataRest[38, idx], gpuGlobalDataRest[39, idx], gpuGlobalDataRest[40, idx])
        #     @cuprintf("\n %.17f %.17f %.17f %.17f", gpuGlobalDataRest[41, idx], gpuGlobalDataRest[42, idx], gpuGlobalDataRest[43, idx], gpuGlobalDataRest[44, idx])
        # end
    end

    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1.0 / det
    # if idx ==3
    #     @cuprintf("\n===Gn===")
    #     @cuprintf("\n %f %f %f %f", sum_1,sum_2,sum_3,sum_4)
    #     @cuprintf("\n %f %f %f %f", sum_5,sum_6,sum_7,sum_8)
    # end
    gpuGlobalDataRest[5, idx] += 2 * (sum_5*sum_delx_sqr - sum_1*sum_delx_dely)*one_by_det
    gpuGlobalDataRest[6, idx] += 2 * (sum_6*sum_delx_sqr - sum_2*sum_delx_dely)*one_by_det
    gpuGlobalDataRest[7, idx] += 2 * (sum_7*sum_delx_sqr - sum_3*sum_delx_dely)*one_by_det
    gpuGlobalDataRest[8, idx] += 2 * (sum_8*sum_delx_sqr - sum_4*sum_delx_dely)*one_by_det
    # if idx ==3
    #     @cuprintf("\n %f %f %f %f", gpuGlobalDataRest[5, idx],gpuGlobalDataRest[6, idx],gpuGlobalDataRest[7, idx],gpuGlobalDataRest[8, idx])
    # end
    return nothing
end