function cal_flux_residual_kernel(gpuGlobalDataCommon, gpuConfigData)
	tx = threadIdx().x
    bx = blockIdx().x - 1
    bw = blockDim().x
	idx = bx * bw + tx
	Gxp1,Gxp2,Gxp3,Gxp4 = 0.0,0.0,0.0,0.0
	Gxn1,Gxn2,Gxn3,Gxn4 = 0.0,0.0,0.0,0.0
	Gyp1,Gyp2,Gyp3,Gyp4 = 0.0,0.0,0.0,0.0
	Gyn1,Gyn2,Gyn3,Gyn4 = 0.0,0.0,0.0,0.0
	if idx > 0 && idx <= gpuGlobalDataCommon[1,end]
		flag1 = gpuGlobalDataCommon[6, idx]
		if flag1 == gpuConfigData[17]
			# if idx == 3
			# 	@cuprintf("\n The value is %f %f %f %f ", Gxp1,  Gxp2,  Gxp3,  Gxp4)
			# 	@cuprintf("\n The value is %f %f %f %f ", Gxn1,  Gxn2,  Gxn3,  Gxn4)
			# 	@cuprintf("\n The value is %f %f %f %f ", Gyp1,  Gyp2,  Gyp3,  Gyp4)
			# 	@cuprintf("\n The value is %f %f %f %f ", Gyn1,  Gyn2,  Gyn3,  Gyn4)
			# end
			@cuda dynamic=true wall_dGx_pos_kernel(gpuGlobalDataCommon, idx, gpuConfigData, Gxp1, Gxp2, Gxp3, Gxp4)
			@cuda dynamic=true wall_dGx_neg_kernel(gpuGlobalDataCommon, idx, gpuConfigData, Gxn1, Gxn2, Gxn3, Gxn4)
			@cuda dynamic=true wall_dGy_neg_kernel(gpuGlobalDataCommon, idx, gpuConfigData, Gyn1, Gyn2, Gyn3, Gyn4)
			CUDAnative.synchronize()
			# if idx == 3
			# 	@cuprintf("\n The value is %f %f %f %f ", Gxp1,  Gxp2,  Gxp3,  Gxp4)
			# 	@cuprintf("\n The value is %f %f %f %f ", Gxn1,  Gxn2,  Gxn3,  Gxn4)
			# 	@cuprintf("\n The value is %f %f %f %f ", Gyp1,  Gyp2,  Gyp3,  Gyp4)
			# 	@cuprintf("\n The value is %f %f %f %f ", Gyn1,  Gyn2,  Gyn3,  Gyn4)
			# end
			gpuGlobalDataCommon[35, idx] = Gxp1 + Gxn1 + Gyn1
			gpuGlobalDataCommon[36, idx] = Gxp2 + Gxn2 + Gyn2
			gpuGlobalDataCommon[37, idx] = Gxp3 + Gxn3 + Gyn3
			gpuGlobalDataCommon[38, idx] = Gxp4 + Gxn4 + Gyn4
		elseif flag1 == gpuConfigData[18]
			@cuda dynamic=true interior_dGx_pos_kernel(gpuGlobalDataCommon, idx, gpuConfigData, Gxp1, Gxp2, Gxp3, Gxp4)
			@cuda dynamic=true interior_dGx_neg_kernel(gpuGlobalDataCommon, idx, gpuConfigData, Gxn1, Gxn2, Gxn3, Gxn4)
			@cuda dynamic=true interior_dGy_pos_kernel(gpuGlobalDataCommon, idx, gpuConfigData, Gyp1, Gyp2, Gyp3, Gyp4)
			@cuda dynamic=true interior_dGy_neg_kernel(gpuGlobalDataCommon, idx, gpuConfigData, Gyn1, Gyn2, Gyn3, Gyn4)
			CUDAnative.synchronize()
			gpuGlobalDataCommon[35, idx] = Gxp1 + Gxn1 + Gyp1 + Gyn1
			gpuGlobalDataCommon[36, idx] = Gxp2 + Gxn2 + Gyp2 + Gyn2
			gpuGlobalDataCommon[37, idx] = Gxp3 + Gxn3 + Gyp3 + Gyn3
			gpuGlobalDataCommon[38, idx] = Gxp4 + Gxn4 + Gyp4 + Gyn4
		elseif flag1 == gpuConfigData[19]
			@cuda dynamic=true outer_dGx_pos_kernel(gpuGlobalDataCommon, idx, gpuConfigData, Gxp1, Gxp2, Gxp3, Gxp4)
			@cuda dynamic=true outer_dGx_neg_kernel(gpuGlobalDataCommon, idx, gpuConfigData, Gxn1, Gxn2, Gxn3, Gxn4)
			@cuda dynamic=true outer_dGy_pos_kernel(gpuGlobalDataCommon, idx, gpuConfigData, Gyp1, Gyp2, Gyp3, Gyp4)
			CUDAnative.synchronize()
			gpuGlobalDataCommon[35, idx] = Gxp1 + Gxn1 + Gyp1
			gpuGlobalDataCommon[36, idx] = Gxp2 + Gxn2 + Gyp2
			gpuGlobalDataCommon[37, idx] = Gxp3 + Gxn3 + Gyp3
			gpuGlobalDataCommon[38, idx] = Gxp4 + Gxn4 + Gyp4
		# else
		# 	@cuprintf("Warning: There is problem with the flux flags %f \n", gpuGlobalDataCommon[6, idx])
		end
	end
	sync_threads()
	return nothing
end
