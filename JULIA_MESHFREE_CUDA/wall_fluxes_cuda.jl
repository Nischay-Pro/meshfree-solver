function wall_dGx_pos_kernel(gpuGlobalDataCommon, idx, gpuConfigData)
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

    qtilde_i = (0,0,0,0)
    qtilde_k = (0,0,0,0)

    for i in 146:173
        gpuGlobalDataCommon[i, idx] = 0.0
    end

    x_i = gpuGlobalDataCommon[2, idx]
    y_i = gpuGlobalDataCommon[3, idx]
    nx = gpuGlobalDataCommon[29, idx]
    ny = gpuGlobalDataCommon[30, idx]

    tx = ny
    ty = -nx

    for iter in 56:75
        conn = Int(gpuGlobalDataCommon[iter, idx])
        if conn == 0
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
        # if idx == 3
        #     @cuprintf("\n %d", conn)
        #     @cuprintf("\n %.17f %.17f %.17f %.17f", qtilde_k[1], qtilde_k[2], qtilde_k[3], qtilde_k[4])
        # end


        if limiter_flag == 1
            venkat_limiter_kernel_i(qtilde_i, gpuGlobalDataCommon, idx, gpuConfigData)
            venkat_limiter_kernel_k(qtilde_k, gpuGlobalDataCommon, conn, gpuConfigData, idx)
            # CUDAnative.synchronize()
            qtilde_i =  (
                            gpuGlobalDataCommon[39, idx] - 0.5*gpuGlobalDataCommon[146,idx]*(delx * gpuGlobalDataCommon[43, idx] + dely * gpuGlobalDataCommon[47, idx]),
                            gpuGlobalDataCommon[40, idx] - 0.5*gpuGlobalDataCommon[147,idx]*(delx * gpuGlobalDataCommon[44, idx] + dely * gpuGlobalDataCommon[48, idx]),
                            gpuGlobalDataCommon[41, idx] - 0.5*gpuGlobalDataCommon[148,idx]*(delx * gpuGlobalDataCommon[45, idx] + dely * gpuGlobalDataCommon[49, idx]),
                            gpuGlobalDataCommon[42, idx] - 0.5*gpuGlobalDataCommon[149,idx]*(delx * gpuGlobalDataCommon[46, idx] + dely * gpuGlobalDataCommon[50, idx])
                        )
            qtilde_k = (
                            gpuGlobalDataCommon[39, conn] - 0.5*gpuGlobalDataCommon[150,idx]*(delx * gpuGlobalDataCommon[43, conn] + dely * gpuGlobalDataCommon[47, conn]),
                            gpuGlobalDataCommon[40, conn] - 0.5*gpuGlobalDataCommon[151,idx]*(delx * gpuGlobalDataCommon[44, conn] + dely * gpuGlobalDataCommon[48, conn]),
                            gpuGlobalDataCommon[41, conn] - 0.5*gpuGlobalDataCommon[152,idx]*(delx * gpuGlobalDataCommon[45, conn] + dely * gpuGlobalDataCommon[49, conn]),
                            gpuGlobalDataCommon[42, conn] - 0.5*gpuGlobalDataCommon[153,idx]*(delx * gpuGlobalDataCommon[46, conn] + dely * gpuGlobalDataCommon[50, conn])
                        )
        end
        # if idx == 3
        #     @cuprintf("\n %d", conn)
        #     @cuprintf("\n %.17f %.17f %.17f %.17f", gpuGlobalDataCommon[146,idx], gpuGlobalDataCommon[147,idx], gpuGlobalDataCommon[148,idx], gpuGlobalDataCommon[149,idx])
        #     @cuprintf("\n %.17f %.17f %.17f %.17f", gpuGlobalDataCommon[150,conn], gpuGlobalDataCommon[151,conn], gpuGlobalDataCommon[152,conn], gpuGlobalDataCommon[153,conn])
        #     @cuprintf("\n %.17f %.17f %.17f %.17f", qtilde_k[1], qtilde_k[2], qtilde_k[3], qtilde_k[4])
        # end
        if limiter_flag == 2
            @cuprintf("\n Havent written the code - die \n")
        end
        # if idx == 3
        #     @cuprintf("\n %d", conn)
        #     @cuprintf("\n %.17f %.17f %.17f %.17f", gpuGlobalDataCommon[170, idx], gpuGlobalDataCommon[171, idx], gpuGlobalDataCommon[172, idx], gpuGlobalDataCommon[173, idx])
        # end
        qtilde_to_primitive_kernel(qtilde_i, gpuConfigData, gpuGlobalDataCommon, idx)
        # if idx == 3
        #     @cuprintf("\n %.17f %.17f %.17f %.17f", gpuGlobalDataCommon[170, idx], gpuGlobalDataCommon[171, idx], gpuGlobalDataCommon[172, idx], gpuGlobalDataCommon[173, idx])
        # end
        flux_quad_GxII_kernel(nx, ny, gpuGlobalDataCommon, idx, 1)
        # if idx == 3
        #     @cuprintf("\n %.17f %.17f %.17f %.17f", gpuGlobalDataCommon[170, idx], gpuGlobalDataCommon[171, idx], gpuGlobalDataCommon[172, idx], gpuGlobalDataCommon[173, idx])
        # end
        qtilde_to_primitive_kernel(qtilde_k, gpuConfigData, gpuGlobalDataCommon, idx)
        # if idx == 3
        #     @cuprintf("\n %.17f %.17f %.17f %.17f", gpuGlobalDataCommon[170, idx], gpuGlobalDataCommon[171, idx], gpuGlobalDataCommon[172, idx], gpuGlobalDataCommon[173, idx])
        # end
        flux_quad_GxII_kernel(nx, ny, gpuGlobalDataCommon, idx, 2)
        # CUDAnative.synchronize()
        gpuGlobalDataCommon[162, idx] += (gpuGlobalDataCommon[158, idx] - gpuGlobalDataCommon[154, idx]) * dels_weights
        gpuGlobalDataCommon[166, idx] += (gpuGlobalDataCommon[158, idx] - gpuGlobalDataCommon[154, idx]) * deln_weights
        gpuGlobalDataCommon[163, idx] += (gpuGlobalDataCommon[159, idx] - gpuGlobalDataCommon[155, idx]) * dels_weights
        gpuGlobalDataCommon[167, idx] += (gpuGlobalDataCommon[159, idx] - gpuGlobalDataCommon[155, idx]) * deln_weights
        gpuGlobalDataCommon[164, idx] += (gpuGlobalDataCommon[160, idx] - gpuGlobalDataCommon[156, idx]) * dels_weights
        gpuGlobalDataCommon[168, idx] += (gpuGlobalDataCommon[160, idx] - gpuGlobalDataCommon[156, idx]) * deln_weights
        gpuGlobalDataCommon[165, idx] += (gpuGlobalDataCommon[161, idx] - gpuGlobalDataCommon[157, idx]) * dels_weights
        gpuGlobalDataCommon[169, idx] += (gpuGlobalDataCommon[161, idx] - gpuGlobalDataCommon[157, idx]) * deln_weights
        # if idx == 3
        #     @cuprintf("\n %d", conn)
        #     @cuprintf("\n %.17f %.17f %.17f %.17f", gpuGlobalDataCommon[170, idx], gpuGlobalDataCommon[171, idx], gpuGlobalDataCommon[172, idx], gpuGlobalDataCommon[173, idx])
        #     @cuprintf("\n %.17f %.17f %.17f %.17f", gpuGlobalDataCommon[154, idx], gpuGlobalDataCommon[155, idx], gpuGlobalDataCommon[156, idx], gpuGlobalDataCommon[157, idx])
        #     @cuprintf("\n %.17f %.17f %.17f %.17f", gpuGlobalDataCommon[158, idx], gpuGlobalDataCommon[159, idx], gpuGlobalDataCommon[160, idx], gpuGlobalDataCommon[161, idx])
        # end
    end

    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1.0 / det
    # if idx ==3
    #     @cuprintf("\n===Gn===")
    #     @cuprintf("\n %.17f %.17f %.17f %.17f", gpuGlobalDataCommon[162, idx],gpuGlobalDataCommon[163, idx],gpuGlobalDataCommon[164, idx],gpuGlobalDataCommon[165, idx])
    #     @cuprintf("\n %.17f %.17f %.17f %.17f", gpuGlobalDataCommon[166, idx],gpuGlobalDataCommon[167, idx],gpuGlobalDataCommon[168, idx],gpuGlobalDataCommon[169, idx])
    # end
    gpuGlobalDataCommon[35, idx] += 2 * (gpuGlobalDataCommon[162, idx]*sum_dely_sqr - gpuGlobalDataCommon[166, idx]*sum_delx_dely)*one_by_det
    gpuGlobalDataCommon[36, idx] += 2 * (gpuGlobalDataCommon[163, idx]*sum_dely_sqr - gpuGlobalDataCommon[167, idx]*sum_delx_dely)*one_by_det
    gpuGlobalDataCommon[37, idx] += 2 * (gpuGlobalDataCommon[164, idx]*sum_dely_sqr - gpuGlobalDataCommon[168, idx]*sum_delx_dely)*one_by_det
    gpuGlobalDataCommon[38, idx] += 2 * (gpuGlobalDataCommon[165, idx]*sum_dely_sqr - gpuGlobalDataCommon[169, idx]*sum_delx_dely)*one_by_det
    # if idx ==3
    #     @cuprintf("\n %.17f %.17f %.17f %.17f", gpuGlobalDataCommon[35, idx],gpuGlobalDataCommon[36, idx],gpuGlobalDataCommon[37, idx],gpuGlobalDataCommon[38, idx])
    # end
    return nothing
end

function wall_dGx_neg_kernel(gpuGlobalDataCommon, idx, gpuConfigData)
    power = gpuConfigData[6]
    limiter_flag = gpuConfigData[7]

    # phi_i1,phi_i2,phi_i3,phi_i4 =  0.0,0.0,0.0,0.0
    # phi_k1,phi_k2,phi_k3,phi_k4 =  0.0,0.0,0.0,0.0
    # G_i1,G_i2,G_i3,G_i4 =  0.0,0.0,0.0,0.0
    # G_k1,G_k2,G_k3,G_k4 =  0.0,0.0,0.0,0.0

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

    # sum_delx_delf1,sum_delx_delf2,sum_delx_delf3,sum_delx_delf4 = 0.0,0.0,0.0,0.0
    # sum_dely_delf1,sum_dely_delf2,sum_dely_delf3,sum_dely_delf4 = 0.0,0.0,0.0,0.0
    qtilde_i = (0,0,0,0)
    qtilde_k = (0,0,0,0)

    for i in 146:173
        gpuGlobalDataCommon[i, idx] = 0.0
    end

    # result1,result2,result3,result4 = 0.0,0.0,0.0,0.0

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
            venkat_limiter_kernel_i(qtilde_i, gpuGlobalDataCommon, idx, gpuConfigData)
            venkat_limiter_kernel_k(qtilde_k, gpuGlobalDataCommon, conn, gpuConfigData, idx)
            # CUDAnative.synchronize()
            qtilde_i =  (
                            gpuGlobalDataCommon[39, idx] - 0.5*gpuGlobalDataCommon[146,idx]*(delx * gpuGlobalDataCommon[43, idx] + dely * gpuGlobalDataCommon[47, idx]),
                            gpuGlobalDataCommon[40, idx] - 0.5*gpuGlobalDataCommon[147,idx]*(delx * gpuGlobalDataCommon[44, idx] + dely * gpuGlobalDataCommon[48, idx]),
                            gpuGlobalDataCommon[41, idx] - 0.5*gpuGlobalDataCommon[148,idx]*(delx * gpuGlobalDataCommon[45, idx] + dely * gpuGlobalDataCommon[49, idx]),
                            gpuGlobalDataCommon[42, idx] - 0.5*gpuGlobalDataCommon[149,idx]*(delx * gpuGlobalDataCommon[46, idx] + dely * gpuGlobalDataCommon[50, idx])
                        )
            qtilde_k = (
                            gpuGlobalDataCommon[39, conn] - 0.5*gpuGlobalDataCommon[150,idx]*(delx * gpuGlobalDataCommon[43, conn] + dely * gpuGlobalDataCommon[47, conn]),
                            gpuGlobalDataCommon[40, conn] - 0.5*gpuGlobalDataCommon[151,idx]*(delx * gpuGlobalDataCommon[44, conn] + dely * gpuGlobalDataCommon[48, conn]),
                            gpuGlobalDataCommon[41, conn] - 0.5*gpuGlobalDataCommon[152,idx]*(delx * gpuGlobalDataCommon[45, conn] + dely * gpuGlobalDataCommon[49, conn]),
                            gpuGlobalDataCommon[42, conn] - 0.5*gpuGlobalDataCommon[153,idx]*(delx * gpuGlobalDataCommon[46, conn] + dely * gpuGlobalDataCommon[50, conn])
                        )
        end

        if limiter_flag == 2
            @cuprintf("\n Havent written the code - die \n")
        end
        qtilde_to_primitive_kernel(qtilde_i, gpuConfigData, gpuGlobalDataCommon, idx)
        flux_quad_GxI_kernel(nx, ny, gpuGlobalDataCommon, idx, 1)
        qtilde_to_primitive_kernel(qtilde_k, gpuConfigData, gpuGlobalDataCommon, idx)
        flux_quad_GxI_kernel(nx, ny, gpuGlobalDataCommon, idx, 2)
        # CUDAnative.synchronize()
        gpuGlobalDataCommon[162, idx] += (gpuGlobalDataCommon[158, idx] - gpuGlobalDataCommon[154, idx]) * dels_weights
        gpuGlobalDataCommon[166, idx] += (gpuGlobalDataCommon[158, idx] - gpuGlobalDataCommon[154, idx]) * deln_weights
        gpuGlobalDataCommon[163, idx] += (gpuGlobalDataCommon[159, idx] - gpuGlobalDataCommon[155, idx]) * dels_weights
        gpuGlobalDataCommon[167, idx] += (gpuGlobalDataCommon[159, idx] - gpuGlobalDataCommon[155, idx]) * deln_weights
        gpuGlobalDataCommon[164, idx] += (gpuGlobalDataCommon[160, idx] - gpuGlobalDataCommon[156, idx]) * dels_weights
        gpuGlobalDataCommon[168, idx] += (gpuGlobalDataCommon[160, idx] - gpuGlobalDataCommon[156, idx]) * deln_weights
        gpuGlobalDataCommon[165, idx] += (gpuGlobalDataCommon[161, idx] - gpuGlobalDataCommon[157, idx]) * dels_weights
        gpuGlobalDataCommon[169, idx] += (gpuGlobalDataCommon[161, idx] - gpuGlobalDataCommon[157, idx]) * deln_weights
        # if idx == 3
        #     @cuprintf("\n *** ")
        #     @cuprintf("\n %f %f %f %f", gpuGlobalDataCommon[154, idx], gpuGlobalDataCommon[155, idx], gpuGlobalDataCommon[156, idx], gpuGlobalDataCommon[157, idx])
        #     @cuprintf("\n %f %f %f %f", gpuGlobalDataCommon[158, idx], gpuGlobalDataCommon[159, idx], gpuGlobalDataCommon[160, idx], gpuGlobalDataCommon[161, idx])
        #     @cuprintf("\n %f", dels_weights)
        #     @cuprintf("\n %f", deln_weights)
        # end
    end

    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1.0 / det

    gpuGlobalDataCommon[35, idx] += 2 * (gpuGlobalDataCommon[162, idx]*sum_dely_sqr - gpuGlobalDataCommon[166, idx]*sum_delx_dely)*one_by_det
    gpuGlobalDataCommon[36, idx] += 2 * (gpuGlobalDataCommon[163, idx]*sum_dely_sqr - gpuGlobalDataCommon[167, idx]*sum_delx_dely)*one_by_det
    gpuGlobalDataCommon[37, idx] += 2 * (gpuGlobalDataCommon[164, idx]*sum_dely_sqr - gpuGlobalDataCommon[168, idx]*sum_delx_dely)*one_by_det
    gpuGlobalDataCommon[38, idx] += 2 * (gpuGlobalDataCommon[165, idx]*sum_dely_sqr - gpuGlobalDataCommon[169, idx]*sum_delx_dely)*one_by_det
    # if idx ==3
    #     @cuprintf("\n %f %f %f %f", gpuGlobalDataCommon[35, idx],gpuGlobalDataCommon[36, idx],gpuGlobalDataCommon[37, idx],gpuGlobalDataCommon[38, idx])
    # end
    return nothing
end

function wall_dGy_neg_kernel(gpuGlobalDataCommon, idx, gpuConfigData)

    power = gpuConfigData[6]
    limiter_flag = gpuConfigData[7]

    # phi_i1,phi_i2,phi_i3,phi_i4 =  0.0,0.0,0.0,0.0
    # phi_k1,phi_k2,phi_k3,phi_k4 =  0.0,0.0,0.0,0.0
    # G_i1,G_i2,G_i3,G_i4 =  0.0,0.0,0.0,0.0
    # G_k1,G_k2,G_k3,G_k4 =  0.0,0.0,0.0,0.0

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

    # sum_delx_delf1,sum_delx_delf2,sum_delx_delf3,sum_delx_delf4 = 0.0,0.0,0.0,0.0
    # sum_dely_delf1,sum_dely_delf2,sum_dely_delf3,sum_dely_delf4 = 0.0,0.0,0.0,0.0
    qtilde_i = (0,0,0,0)
    qtilde_k = (0,0,0,0)

    for i in 146:173
        gpuGlobalDataCommon[i, idx] = 0.0
    end

    # result1,result2,result3,result4 = 0.0,0.0,0.0,0.0

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
            venkat_limiter_kernel_i(qtilde_i, gpuGlobalDataCommon, idx, gpuConfigData)
            venkat_limiter_kernel_k(qtilde_k, gpuGlobalDataCommon, conn, gpuConfigData, idx)
            # CUDAnative.synchronize()
            qtilde_i =  (
                            gpuGlobalDataCommon[39, idx] - 0.5*gpuGlobalDataCommon[146,idx]*(delx * gpuGlobalDataCommon[43, idx] + dely * gpuGlobalDataCommon[47, idx]),
                            gpuGlobalDataCommon[40, idx] - 0.5*gpuGlobalDataCommon[147,idx]*(delx * gpuGlobalDataCommon[44, idx] + dely * gpuGlobalDataCommon[48, idx]),
                            gpuGlobalDataCommon[41, idx] - 0.5*gpuGlobalDataCommon[148,idx]*(delx * gpuGlobalDataCommon[45, idx] + dely * gpuGlobalDataCommon[49, idx]),
                            gpuGlobalDataCommon[42, idx] - 0.5*gpuGlobalDataCommon[149,idx]*(delx * gpuGlobalDataCommon[46, idx] + dely * gpuGlobalDataCommon[50, idx])
                        )
            qtilde_k = (
                            gpuGlobalDataCommon[39, conn] - 0.5*gpuGlobalDataCommon[150,idx]*(delx * gpuGlobalDataCommon[43, conn] + dely * gpuGlobalDataCommon[47, conn]),
                            gpuGlobalDataCommon[40, conn] - 0.5*gpuGlobalDataCommon[151,idx]*(delx * gpuGlobalDataCommon[44, conn] + dely * gpuGlobalDataCommon[48, conn]),
                            gpuGlobalDataCommon[41, conn] - 0.5*gpuGlobalDataCommon[152,idx]*(delx * gpuGlobalDataCommon[45, conn] + dely * gpuGlobalDataCommon[49, conn]),
                            gpuGlobalDataCommon[42, conn] - 0.5*gpuGlobalDataCommon[153,idx]*(delx * gpuGlobalDataCommon[46, conn] + dely * gpuGlobalDataCommon[50, conn])
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
        # if idx == 3
        #     @cuprintf("\n %d", conn)
        #     @cuprintf("\n %.17f %.17f %.17f %.17f", gpuGlobalDataCommon[170, idx], gpuGlobalDataCommon[171, idx], gpuGlobalDataCommon[172, idx], gpuGlobalDataCommon[173, idx])
        #     @cuprintf("\n %.17f %.17f %.17f %.17f", gpuGlobalDataCommon[154, idx], gpuGlobalDataCommon[155, idx], gpuGlobalDataCommon[156, idx], gpuGlobalDataCommon[157, idx])
        #     @cuprintf("\n %.17f %.17f %.17f %.17f", gpuGlobalDataCommon[158, idx], gpuGlobalDataCommon[159, idx], gpuGlobalDataCommon[160, idx], gpuGlobalDataCommon[161, idx])
        # end
    end

    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1.0 / det
    # if idx ==3
    #     @cuprintf("\n===Gn===")
    #     @cuprintf("\n %f %f %f %f", gpuGlobalDataCommon[162, idx],gpuGlobalDataCommon[163, idx],gpuGlobalDataCommon[164, idx],gpuGlobalDataCommon[165, idx])
    #     @cuprintf("\n %f %f %f %f", gpuGlobalDataCommon[166, idx],gpuGlobalDataCommon[167, idx],gpuGlobalDataCommon[168, idx],gpuGlobalDataCommon[169, idx])
    # end
    gpuGlobalDataCommon[35, idx] += 2 * (gpuGlobalDataCommon[166, idx]*sum_delx_sqr - gpuGlobalDataCommon[162, idx]*sum_delx_dely)*one_by_det
    gpuGlobalDataCommon[36, idx] += 2 * (gpuGlobalDataCommon[167, idx]*sum_delx_sqr - gpuGlobalDataCommon[163, idx]*sum_delx_dely)*one_by_det
    gpuGlobalDataCommon[37, idx] += 2 * (gpuGlobalDataCommon[168, idx]*sum_delx_sqr - gpuGlobalDataCommon[164, idx]*sum_delx_dely)*one_by_det
    gpuGlobalDataCommon[38, idx] += 2 * (gpuGlobalDataCommon[169, idx]*sum_delx_sqr - gpuGlobalDataCommon[165, idx]*sum_delx_dely)*one_by_det
    # if idx ==3
    #     @cuprintf("\n %f %f %f %f", gpuGlobalDataCommon[35, idx],gpuGlobalDataCommon[36, idx],gpuGlobalDataCommon[37, idx],gpuGlobalDataCommon[38, idx])
    # end
    return nothing
end