function wall_dGx_pos_kernel(gpuGlobalDataCommon, idx, gpuConfigData, Gxp)
    power = gpuConfigData[6]
    limiter_flag = gpuConfigData[7]

    sum_delx_sqr = 0.0
    sum_dely_sqr = 0.0
    sum_delx_dely = 0.0

    sum_delx_delf = (0,0,0,0)
    sum_dely_delf = (0,0,0,0)
    qtilde_i = (0,0,0,0)
    qtilde_k = (0,0,0,0)
    phi_i = (0,0,0,0)
    phi_k = (0,0,0,0)
    result = (0,0,0,0)
    G_i = (0,0,0,0)
    G_k = (0,0,0,0)

    x_i = gpuGlobalDataCommon[2]
    y_i = gpuGlobalDataCommon[3]
    nx = gpuGlobalDataCommon[29]
    ny = gpuGlobalDataCommon[30]

    tx = ny
    ty = -nx

end

function wall_dGx_neg_kernel(gpuGlobaldata, idx, gpuConfigData)

end

function wall_dGy_neg_kernel(gpuGlobaldata, idx, gpuConfigData)

end