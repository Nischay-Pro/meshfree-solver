function wall_dGx_pos_kernel(gpuGlobalDataCommon, idx, gpuConfigData, Gxp)
    power = gpuConfigData[6]
    limiter_flag = gpuConfigData[7]

    sum_delx_sqr = 0.0
    sum_dely_sqr = 0.0
    sum_delx_dely = 0.0

    sum_delx_delf = (0,0,0,0)
    sum_dely_delf = (0,0,0,0)
end

function wall_dGx_neg_kernel(gpuGlobaldata, idx, gpuConfigData)

end

function wall_dGy_neg_kernel(gpuGlobaldata, idx, gpuConfigData)

end