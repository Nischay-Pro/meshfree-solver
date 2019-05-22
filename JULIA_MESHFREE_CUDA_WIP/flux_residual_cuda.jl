function cal_flux_residual_kernel(gpuGlobalDataCommon, gpuConfigData, Gxp, Gxn, Gyp, Gyn, phi_i, phi_k, G_i, G_k,sum_delx_delf, sum_dely_delf)
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
			@cuda dynamic=true wall_dGx_pos_kernel(gpuGlobalDataCommon, idx, gpuConfigData, Gxp, phi_i, phi_k, G_i, G_k, sum_delx_delf, sum_dely_delf)
			# @cuda dynamic=true wall_dGx_neg_kernel(gpuGlobalDataCommon, idx, gpuConfigData, Gxn, phi_i, phi_k, G_i, G_k, sum_delx_delf, sum_dely_delf)
			# @cuda dynamic=true wall_dGy_neg_kernel(gpuGlobalDataCommon, idx, gpuConfigData, Gyn, phi_i, phi_k, G_i, G_k, sum_delx_delf, sum_dely_delf)
			CUDAnative.synchronize()
			# @cuprintf("The value is %f", Gxp)
			gpuGlobalDataCommon[35, idx] = Gxp[1] + Gxn[1] + Gyp[1] + Gyn[1]
			gpuGlobalDataCommon[36, idx] = Gxp[2] + Gxn[2] + Gyp[2] + Gyn[2]
			gpuGlobalDataCommon[37, idx] = Gxp[3] + Gxn[3] + Gyp[3] + Gyn[3]
			gpuGlobalDataCommon[38, idx] = Gxp[4] + Gxn[4] + Gyp[4] + Gyn[4]
		elseif flag1 == gpuConfigData[18]
			# @cuda dynamic=true outer_dGx_pos_kernel(gpuGlobalDataCommon, idx, gpuConfigData, Gxp, phi_i, phi_k, G_i, G_k, sum_delx_delf, sum_dely_delf)
			# CUDAnative.synchronize()
			gpuGlobalDataCommon[35, idx] = Gxp[1] + Gxn[1] + Gyp[1]
			gpuGlobalDataCommon[36, idx] = Gxp[2] + Gxn[2] + Gyp[2]
			gpuGlobalDataCommon[37, idx] = Gxp[3] + Gxn[3] + Gyp[3]
			gpuGlobalDataCommon[38, idx] = Gxp[4] + Gxn[4] + Gyp[4]
		elseif flag1 == gpuConfigData[19]
			gpuGlobalDataCommon[35, idx] = Gxp[1] + Gxn[1] + Gyp[1] + Gyn[1]
			gpuGlobalDataCommon[36, idx] = Gxp[2] + Gxn[2] + Gyp[2] + Gyn[2]
			gpuGlobalDataCommon[37, idx] = Gxp[3] + Gxn[3] + Gyp[3] + Gyn[3]
			gpuGlobalDataCommon[38, idx] = Gxp[4] + Gxn[4] + Gyp[4] + Gyn[4]
		else
			@cuprintf("Warning: There is problem with the flux flags %f \n", gpuGlobalDataCommon[6, idx])
		end
	end
	return
end
