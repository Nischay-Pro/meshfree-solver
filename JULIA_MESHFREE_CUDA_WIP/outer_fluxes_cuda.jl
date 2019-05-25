function outer_dGx_pos_kernel(gpuGlobalDataCommon, idx, gpuConfigData, Gxp1, Gxp2, Gxp3, Gxp4)

    power = gpuConfigData[6]
    limiter_flag = gpuConfigData[7]

    phi_i1,phi_i2,phi_i3,phi_i4 =  0.0,0.0,0.0,0.0
    phi_k1,phi_k2,phi_k3,phi_k4 =  0.0,0.0,0.0,0.0
    G_i1,G_i2,G_i3,G_i4 =  0.0,0.0,0.0,0.0
    G_k1,G_k2,G_k3,G_k4 =  0.0,0.0,0.0,0.0

    sum_delx_sqr = 0.0
    sum_dely_sqr = 0.0
    sum_delx_dely = 0.0

    sum_delx_delf1,sum_delx_delf2,sum_delx_delf3,sum_delx_delf4 = 0.0,0.0,0.0,0.0
    sum_dely_delf1,sum_dely_delf2,sum_dely_delf3,sum_dely_delf4 = 0.0,0.0,0.0,0.0
    qtilde_i = (0,0,0,0)
    qtilde_k = (0,0,0,0)

    result1,result2,result3,result4 = 0.0,0.0,0.0,0.0

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
            venkat_limiter_kernel(qtilde_i, gpuGlobalDataCommon, idx, gpuConfigData, phi_i1,phi_i2, phi_i3,phi_i4)
            venkat_limiter_kernel(qtilde_k, gpuGlobalDataCommon, conn, gpuConfigData, phi_k1,phi_k2,phi_k3,phi_k4)
            # CUDAnative.synchronize()
            qtilde_i =  (
                            gpuGlobalDataCommon[39, idx] - 0.5*phi_i1*(delx * gpuGlobalDataCommon[43, idx] + dely * gpuGlobalDataCommon[47, idx]),
                            gpuGlobalDataCommon[40, idx] - 0.5*phi_i2*(delx * gpuGlobalDataCommon[44, idx] + dely * gpuGlobalDataCommon[48, idx]),
                            gpuGlobalDataCommon[41, idx] - 0.5*phi_i3*(delx * gpuGlobalDataCommon[45, idx] + dely * gpuGlobalDataCommon[49, idx]),
                            gpuGlobalDataCommon[42, idx] - 0.5*phi_i4*(delx * gpuGlobalDataCommon[46, idx] + dely * gpuGlobalDataCommon[50, idx])
                        )
            qtilde_k = (
                            gpuGlobalDataCommon[39, conn] - 0.5*phi_k1*(delx * gpuGlobalDataCommon[43, conn] + dely * gpuGlobalDataCommon[47, conn]),
                            gpuGlobalDataCommon[40, conn] - 0.5*phi_k2*(delx * gpuGlobalDataCommon[44, conn] + dely * gpuGlobalDataCommon[48, conn]),
                            gpuGlobalDataCommon[41, conn] - 0.5*phi_k3*(delx * gpuGlobalDataCommon[45, conn] + dely * gpuGlobalDataCommon[49, conn]),
                            gpuGlobalDataCommon[42, conn] - 0.5*phi_k4*(delx * gpuGlobalDataCommon[46, conn] + dely * gpuGlobalDataCommon[50, conn])
                        )
        end

        if limiter_flag == 2
            @cuprintf("\n Havent written the code - die \n")
        end

        qtilde_to_primitive_kernel(qtilde_i, gpuConfigData, result1,result2,result3,result4)
        flux_quad_GxIII_kernel(nx, ny, result1,result2,result3,result4, G_i1,G_i2,G_i3,G_i4)
        qtilde_to_primitive_kernel(qtilde_k, gpuConfigData, result1,result2,result3,result4)
        flux_quad_GxIII_kernel(nx, ny, result1,result2,result3,result4, G_i1,G_i2,G_i3,G_i4)
        # CUDAnative.synchronize()
        sum_delx_delf1 += (G_k1 - G_i1) * dels_weights
        sum_dely_delf1 += (G_k1 - G_i1) * deln_weights
        sum_delx_delf2 += (G_k2 - G_i2) * dels_weights
        sum_dely_delf2 += (G_k2 - G_i2) * deln_weights
        sum_delx_delf3 += (G_k3 - G_i3) * dels_weights
        sum_dely_delf3 += (G_k3 - G_i3) * deln_weights
        sum_delx_delf4 += (G_k4 - G_i4) * dels_weights
        sum_dely_delf4 += (G_k4 - G_i4) * deln_weights
    end

    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1.0 / det
    Gxp1 = (sum_delx_delf1*sum_dely_sqr - sum_dely_delf1*sum_delx_dely)*one_by_det
    Gxp2 = (sum_delx_delf2*sum_dely_sqr - sum_dely_delf2*sum_delx_dely)*one_by_det
    Gxp3 = (sum_delx_delf3*sum_dely_sqr - sum_dely_delf3*sum_delx_dely)*one_by_det
    Gxp4 = (sum_delx_delf4*sum_dely_sqr - sum_dely_delf4*sum_delx_dely)*one_by_det
    return nothing
end

function outer_dGx_neg_kernel(gpuGlobalDataCommon, idx, gpuConfigData, Gxn1, Gxn2, Gxn3, Gxn4)

    power = gpuConfigData[6]
    limiter_flag = gpuConfigData[7]

    phi_i1,phi_i2,phi_i3,phi_i4 =  0.0,0.0,0.0,0.0
    phi_k1,phi_k2,phi_k3,phi_k4 =  0.0,0.0,0.0,0.0
    G_i1,G_i2,G_i3,G_i4 =  0.0,0.0,0.0,0.0
    G_k1,G_k2,G_k3,G_k4 =  0.0,0.0,0.0,0.0

    sum_delx_sqr = 0.0
    sum_dely_sqr = 0.0
    sum_delx_dely = 0.0

    sum_delx_delf1,sum_delx_delf2,sum_delx_delf3,sum_delx_delf4 = 0.0,0.0,0.0,0.0
    sum_dely_delf1,sum_dely_delf2,sum_dely_delf3,sum_dely_delf4 = 0.0,0.0,0.0,0.0
    qtilde_i = (0,0,0,0)
    qtilde_k = (0,0,0,0)

    result1,result2,result3,result4 = 0.0,0.0,0.0,0.0

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
            venkat_limiter_kernel(qtilde_i, gpuGlobalDataCommon, idx, gpuConfigData, phi_i1,phi_i2, phi_i3,phi_i4)
            venkat_limiter_kernel(qtilde_k, gpuGlobalDataCommon, conn, gpuConfigData, phi_k1,phi_k2,phi_k3,phi_k4)
            # CUDAnative.synchronize()
            qtilde_i =  (
                            gpuGlobalDataCommon[39, idx] - 0.5*phi_i1*(delx * gpuGlobalDataCommon[43, idx] + dely * gpuGlobalDataCommon[47, idx]),
                            gpuGlobalDataCommon[40, idx] - 0.5*phi_i2*(delx * gpuGlobalDataCommon[44, idx] + dely * gpuGlobalDataCommon[48, idx]),
                            gpuGlobalDataCommon[41, idx] - 0.5*phi_i3*(delx * gpuGlobalDataCommon[45, idx] + dely * gpuGlobalDataCommon[49, idx]),
                            gpuGlobalDataCommon[42, idx] - 0.5*phi_i4*(delx * gpuGlobalDataCommon[46, idx] + dely * gpuGlobalDataCommon[50, idx])
                        )
            qtilde_k = (
                            gpuGlobalDataCommon[39, conn] - 0.5*phi_k1*(delx * gpuGlobalDataCommon[43, conn] + dely * gpuGlobalDataCommon[47, conn]),
                            gpuGlobalDataCommon[40, conn] - 0.5*phi_k2*(delx * gpuGlobalDataCommon[44, conn] + dely * gpuGlobalDataCommon[48, conn]),
                            gpuGlobalDataCommon[41, conn] - 0.5*phi_k3*(delx * gpuGlobalDataCommon[45, conn] + dely * gpuGlobalDataCommon[49, conn]),
                            gpuGlobalDataCommon[42, conn] - 0.5*phi_k4*(delx * gpuGlobalDataCommon[46, conn] + dely * gpuGlobalDataCommon[50, conn])
                        )
        end

        if limiter_flag == 2
            @cuprintf("\n Havent written the code - die \n")
        end

        qtilde_to_primitive_kernel(qtilde_i, gpuConfigData, result1,result2,result3,result4)
        flux_quad_GxIV_kernel(nx, ny, result1,result2,result3,result4, G_i1,G_i2,G_i3,G_i4)
        qtilde_to_primitive_kernel(qtilde_k, gpuConfigData, result1,result2,result3,result4)
        flux_quad_GxIV_kernel(nx, ny, result1,result2,result3,result4, G_i1,G_i2,G_i3,G_i4)
        # CUDAnative.synchronize()
        sum_delx_delf1 += (G_k1 - G_i1) * dels_weights
        sum_dely_delf1 += (G_k1 - G_i1) * deln_weights
        sum_delx_delf2 += (G_k2 - G_i2) * dels_weights
        sum_dely_delf2 += (G_k2 - G_i2) * deln_weights
        sum_delx_delf3 += (G_k3 - G_i3) * dels_weights
        sum_dely_delf3 += (G_k3 - G_i3) * deln_weights
        sum_delx_delf4 += (G_k4 - G_i4) * dels_weights
        sum_dely_delf4 += (G_k4 - G_i4) * deln_weights
    end

    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1.0 / det
    Gxn1 = (sum_delx_delf1*sum_dely_sqr - sum_dely_delf1*sum_delx_dely)*one_by_det
    Gxn2 = (sum_delx_delf2*sum_dely_sqr - sum_dely_delf2*sum_delx_dely)*one_by_det
    Gxn3 = (sum_delx_delf3*sum_dely_sqr - sum_dely_delf3*sum_delx_dely)*one_by_det
    Gxn4 = (sum_delx_delf4*sum_dely_sqr - sum_dely_delf4*sum_delx_dely)*one_by_det
    return nothing
end

function outer_dGy_pos_kernel(gpuGlobalDataCommon, idx, gpuConfigData, Gyp1, Gyp2, Gyp3, Gyp4)

    power = gpuConfigData[6]
    limiter_flag = gpuConfigData[7]

    phi_i1,phi_i2,phi_i3,phi_i4 =  0.0,0.0,0.0,0.0
    phi_k1,phi_k2,phi_k3,phi_k4 =  0.0,0.0,0.0,0.0
    G_i1,G_i2,G_i3,G_i4 =  0.0,0.0,0.0,0.0
    G_k1,G_k2,G_k3,G_k4 =  0.0,0.0,0.0,0.0

    sum_delx_sqr = 0.0
    sum_dely_sqr = 0.0
    sum_delx_dely = 0.0

    sum_delx_delf1,sum_delx_delf2,sum_delx_delf3,sum_delx_delf4 = 0.0,0.0,0.0,0.0
    sum_dely_delf1,sum_dely_delf2,sum_dely_delf3,sum_dely_delf4 = 0.0,0.0,0.0,0.0
    qtilde_i = (0,0,0,0)
    qtilde_k = (0,0,0,0)

    result1,result2,result3,result4 = 0.0,0.0,0.0,0.0

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
            venkat_limiter_kernel(qtilde_i, gpuGlobalDataCommon, idx, gpuConfigData, phi_i1,phi_i2, phi_i3,phi_i4)
            venkat_limiter_kernel(qtilde_k, gpuGlobalDataCommon, conn, gpuConfigData, phi_k1,phi_k2,phi_k3,phi_k4)
            # CUDAnative.synchronize()
            qtilde_i =  (
                            gpuGlobalDataCommon[39, idx] - 0.5*phi_i1*(delx * gpuGlobalDataCommon[43, idx] + dely * gpuGlobalDataCommon[47, idx]),
                            gpuGlobalDataCommon[40, idx] - 0.5*phi_i2*(delx * gpuGlobalDataCommon[44, idx] + dely * gpuGlobalDataCommon[48, idx]),
                            gpuGlobalDataCommon[41, idx] - 0.5*phi_i3*(delx * gpuGlobalDataCommon[45, idx] + dely * gpuGlobalDataCommon[49, idx]),
                            gpuGlobalDataCommon[42, idx] - 0.5*phi_i4*(delx * gpuGlobalDataCommon[46, idx] + dely * gpuGlobalDataCommon[50, idx])
                        )
            qtilde_k = (
                            gpuGlobalDataCommon[39, conn] - 0.5*phi_k1*(delx * gpuGlobalDataCommon[43, conn] + dely * gpuGlobalDataCommon[47, conn]),
                            gpuGlobalDataCommon[40, conn] - 0.5*phi_k2*(delx * gpuGlobalDataCommon[44, conn] + dely * gpuGlobalDataCommon[48, conn]),
                            gpuGlobalDataCommon[41, conn] - 0.5*phi_k3*(delx * gpuGlobalDataCommon[45, conn] + dely * gpuGlobalDataCommon[49, conn]),
                            gpuGlobalDataCommon[42, conn] - 0.5*phi_k4*(delx * gpuGlobalDataCommon[46, conn] + dely * gpuGlobalDataCommon[50, conn])
                        )
        end

        if limiter_flag == 2
            @cuprintf("\n Havent written the code - die \n")
        end

        qtilde_to_primitive_kernel(qtilde_i, gpuConfigData, result1,result2,result3,result4)
        flux_Gyp_kernel(nx, ny, result1,result2,result3,result4, G_i1,G_i2,G_i3,G_i4)
        qtilde_to_primitive_kernel(qtilde_k, gpuConfigData, result1,result2,result3,result4)
        flux_Gyp_kernel(nx, ny, result1,result2,result3,result4, G_i1,G_i2,G_i3,G_i4)
        # CUDAnative.synchronize()
        sum_delx_delf1 += (G_k1 - G_i1) * dels_weights
        sum_dely_delf1 += (G_k1 - G_i1) * deln_weights
        sum_delx_delf2 += (G_k2 - G_i2) * dels_weights
        sum_dely_delf2 += (G_k2 - G_i2) * deln_weights
        sum_delx_delf3 += (G_k3 - G_i3) * dels_weights
        sum_dely_delf3 += (G_k3 - G_i3) * deln_weights
        sum_delx_delf4 += (G_k4 - G_i4) * dels_weights
        sum_dely_delf4 += (G_k4 - G_i4) * deln_weights
    end

    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1.0 / det
    Gyp1 = (sum_delx_delf1*sum_dely_sqr - sum_dely_delf1*sum_delx_dely)*one_by_det
    Gyp2 = (sum_delx_delf2*sum_dely_sqr - sum_dely_delf2*sum_delx_dely)*one_by_det
    Gyp3 = (sum_delx_delf3*sum_dely_sqr - sum_dely_delf3*sum_delx_dely)*one_by_det
    Gyp4 = (sum_delx_delf4*sum_dely_sqr - sum_dely_delf4*sum_delx_dely)*one_by_det
    return nothing
end