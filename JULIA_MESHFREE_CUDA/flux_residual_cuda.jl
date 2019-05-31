@inline function cal_flux_residual_kernel(gpuGlobalDataCommon, gpuConfigData)
	tx = threadIdx().x
    bx = blockIdx().x - 1
    bw = blockDim().x
	idx = bx * bw + tx

	if idx > 0 && idx <= gpuGlobalDataCommon[1,end]
		flag1 = gpuGlobalDataCommon[6, idx]
		if flag1 == gpuConfigData[17]
			gpuGlobalDataCommon[35, idx] = 0.0
			gpuGlobalDataCommon[36, idx] = 0.0
			gpuGlobalDataCommon[37, idx] = 0.0
			gpuGlobalDataCommon[38, idx] = 0.0
			wall_dGx_pos_kernel(gpuGlobalDataCommon, idx, gpuConfigData)
			# sync_threads()
			wall_dGx_neg_kernel(gpuGlobalDataCommon, idx, gpuConfigData)
			# sync_threads()
			wall_dGy_neg_kernel(gpuGlobalDataCommon, idx, gpuConfigData)
			# sync_threads()
			# gpuGlobalDataCommon[35, idx] = 2 * gpuGlobalDataCommon[35, idx]
			# gpuGlobalDataCommon[36, idx] = 2 * gpuGlobalDataCommon[36, idx]
			# gpuGlobalDataCommon[37, idx] = 2 * gpuGlobalDataCommon[37, idx]
			# gpuGlobalDataCommon[38, idx] = 2 * gpuGlobalDataCommon[38, idx]
		elseif flag1 == gpuConfigData[18]
			gpuGlobalDataCommon[35, idx] = 0.0
			gpuGlobalDataCommon[36, idx] = 0.0
			gpuGlobalDataCommon[37, idx] = 0.0
			gpuGlobalDataCommon[38, idx] = 0.0
			interior_dGx_pos_kernel(gpuGlobalDataCommon, idx, gpuConfigData)
			# sync_threads()
			interior_dGx_neg_kernel(gpuGlobalDataCommon, idx, gpuConfigData)
			# sync_threads()
			interior_dGy_pos_kernel(gpuGlobalDataCommon, idx, gpuConfigData)
			# sync_threads()
			interior_dGy_neg_kernel(gpuGlobalDataCommon, idx, gpuConfigData)
			# sync_threads()
		elseif flag1 == gpuConfigData[19]
			gpuGlobalDataCommon[35, idx] = 0.0
			gpuGlobalDataCommon[36, idx] = 0.0
			gpuGlobalDataCommon[37, idx] = 0.0
			gpuGlobalDataCommon[38, idx] = 0.0
			outer_dGx_pos_kernel(gpuGlobalDataCommon, idx, gpuConfigData)
			# sync_threads()
			outer_dGx_neg_kernel(gpuGlobalDataCommon, idx, gpuConfigData)
			# sync_threads()
			outer_dGy_pos_kernel(gpuGlobalDataCommon, idx, gpuConfigData)
			# sync_threads()
		# else
		# 	@cuprintf("Warning: There is problem with the flux flags %f \n", gpuGlobalDataCommon[6, idx])
		end
	end
	sync_threads()
	return nothing
end
