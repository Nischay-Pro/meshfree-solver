function cal_flux_residual_kernel(gpuGlobalDataCommon, gpuConfigData, Gxp, Gxn, Gyp, Gyn)
	tx = threadIdx().x
    bx = blockIdx().x - 1
    bw = blockDim().x
	idx = bx * bw + tx
	fill!(Gxp, 0.0)
	fill!(Gxn, 0.0)
	fill!(Gyp, 0.0)
	fill!(Gxn, 0.0)
	if idx > 0 && idx <= gpuGlobalDataCommon[1,end]
		flag1 = gpuGlobalDataCommon[6, idx]
		if flag1 == gpuConfigData[17]
			wall_dGx_pos_kernel(gpuGlobalDataCommon, idx, gpuConfigData, Gxp)
			gpuGlobalDataCommon[35] = Gxp[1] + Gxn[1] + Gyp[1] + Gyn[1]
			gpuGlobalDataCommon[36] = Gxp[2] + Gxn[2] + Gyp[2] + Gyn[2]
			gpuGlobalDataCommon[37] = Gxp[3] + Gxn[3] + Gyp[3] + Gyn[3]
			gpuGlobalDataCommon[38] = Gxp[4] + Gxn[4] + Gyp[4] + Gyn[4]
		elseif flag1 == gpuConfigData[18]
			gpuGlobalDataCommon[35] = Gxp[1] + Gxn[1] + Gyp[1]
			gpuGlobalDataCommon[36] = Gxp[2] + Gxn[2] + Gyp[2]
			gpuGlobalDataCommon[37] = Gxp[3] + Gxn[3] + Gyp[3]
			gpuGlobalDataCommon[38] = Gxp[4] + Gxn[4] + Gyp[4]
		elseif flag1 == gpuConfigData[19]
			gpuGlobalDataCommon[35] = Gxp[1] + Gxn[1] + Gyp[1] + Gyn[1]
			gpuGlobalDataCommon[36] = Gxp[2] + Gxn[2] + Gyp[2] + Gyn[2]
			gpuGlobalDataCommon[37] = Gxp[3] + Gxn[3] + Gyp[3] + Gyn[3]
			gpuGlobalDataCommon[38] = Gxp[4] + Gxn[4] + Gyp[4] + Gyn[4]
		else
			@cuprintf("Warning: There is problem with the flux flags %f \n", gpuGlobalDataCommon[6, idx])
		end
	end
	return
end
