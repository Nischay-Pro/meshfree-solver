function cal_flux_residual_kernel(gpuGlobalDataCommon, gpuGlobalDataFixedPoint, gpuConfigData)
	tx = threadIdx().x
    bx = blockIdx().x - 1
    bw = blockDim().x
	idx = bx * bw + tx

	if idx > 0 && idx <= gpuGlobalDataCommon[1,end]
		flag1 = gpuGlobalDataFixedPoint[idx].flag_1
		if flag1 == gpuConfigData[17]
			wall_kernel(gpuGlobalDataCommon, gpuGlobalDataFixedPoint, idx, gpuConfigData)
		elseif flag1 == gpuConfigData[18]
			interior_kernel(gpuGlobalDataCommon, gpuGlobalDataFixedPoint, idx, gpuConfigData)
			# sync_threads()
		elseif flag1 == gpuConfigData[19]
			outer_kernel(gpuGlobalDataCommon, gpuGlobalDataFixedPoint, idx, gpuConfigData)
			# sync_threads()
		# else
		# 	@cuprintf("Warning: There is problem with the flux flags %f \n", gpuGlobalDataFixedPoint[idx].flag_1)
		end
	end
	# sync_threads()
	return nothing
end

@inline function wall_kernel(gpuGlobalDataCommon, gpuGlobalDataFixedPoint, idx, gpuConfigData)
	gpuGlobalDataCommon[35, idx] = 0.0
	gpuGlobalDataCommon[36, idx] = 0.0
	gpuGlobalDataCommon[37, idx] = 0.0
	gpuGlobalDataCommon[38, idx] = 0.0
	wall_dGx_pos_kernel(gpuGlobalDataCommon, gpuGlobalDataFixedPoint, idx, gpuConfigData)
	# sync_threads()
	wall_dGx_neg_kernel(gpuGlobalDataCommon, gpuGlobalDataFixedPoint, idx, gpuConfigData)
	# sync_threads()
	wall_dGy_neg_kernel(gpuGlobalDataCommon, gpuGlobalDataFixedPoint, idx, gpuConfigData)
	return nothing
end

@inline function interior_kernel(gpuGlobalDataCommon, gpuGlobalDataFixedPoint, idx, gpuConfigData)
	gpuGlobalDataCommon[35, idx] = 0.0
	gpuGlobalDataCommon[36, idx] = 0.0
	gpuGlobalDataCommon[37, idx] = 0.0
	gpuGlobalDataCommon[38, idx] = 0.0
	interior_dGx_pos_kernel(gpuGlobalDataCommon, gpuGlobalDataFixedPoint, idx, gpuConfigData)
	# sync_threads()
	interior_dGx_neg_kernel(gpuGlobalDataCommon, gpuGlobalDataFixedPoint, idx, gpuConfigData)
	# sync_threads()
	interior_dGy_pos_kernel(gpuGlobalDataCommon, gpuGlobalDataFixedPoint, idx, gpuConfigData)
	# sync_threads()
	interior_dGy_neg_kernel(gpuGlobalDataCommon, gpuGlobalDataFixedPoint, idx, gpuConfigData)
	return nothing
end

@inline function outer_kernel(gpuGlobalDataCommon, gpuGlobalDataFixedPoint, idx, gpuConfigData)
	gpuGlobalDataCommon[35, idx] = 0.0
	gpuGlobalDataCommon[36, idx] = 0.0
	gpuGlobalDataCommon[37, idx] = 0.0
	gpuGlobalDataCommon[38, idx] = 0.0
	outer_dGx_pos_kernel(gpuGlobalDataCommon, gpuGlobalDataFixedPoint, idx, gpuConfigData)
	# sync_threads()
	outer_dGx_neg_kernel(gpuGlobalDataCommon, gpuGlobalDataFixedPoint, idx, gpuConfigData)
	# sync_threads()
	outer_dGy_pos_kernel(gpuGlobalDataCommon, gpuGlobalDataFixedPoint, idx, gpuConfigData)
	return nothing
end