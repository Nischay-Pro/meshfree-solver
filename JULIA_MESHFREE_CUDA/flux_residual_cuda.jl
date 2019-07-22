function cal_flux_residual_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, gpuConfigData, numPoints)
	tx = threadIdx().x
    bx = blockIdx().x - 1
    bw = blockDim().x
	idx = bx * bw + tx
	temp = @SVector zeros(4)

	shared = @cuStaticSharedMem(Float64, 512)
	if idx > 0 && idx <= numPoints
		if idx <= numPoints
			flag1 = gpuGlobalDataFixedPoint[idx].flag_1

		gpuGlobalDataRest[5, idx] = 0.0
		gpuGlobalDataRest[6, idx] = 0.0
		gpuGlobalDataRest[7, idx] = 0.0
		gpuGlobalDataRest[8, idx] = 0.0

		for i in 1:4
            gpuGlobalDataRest[20+i, idx] = gpuGlobalDataRest[8+i, idx]
            gpuGlobalDataRest[24+i, idx] = gpuGlobalDataRest[8+i, idx]
            for iter in 5:14
                conn = gpuGlobalDataConn[iter, idx]
                if conn == 0.0
                    break
                end
                update_q(gpuGlobalDataRest, idx, i, conn)
            end
		end

		# elseif idx <= 2 * numPoints
		# 	flag1 = gpuGlobalDataFixedPoint[idx - numPoints].flag_1
		# elseif idx <= 3 * numPoints
		# 	flag1 = gpuGlobalDataFixedPoint[idx - 2 * numPoints].flag_1
		# else
		# 	flag1 = gpuGlobalDataFixedPoint[idx - 3 * numPoints].flag_1
		end
		power = gpuConfigData[6]
    	limiter_flag = gpuConfigData[7]
    	gamma = gpuConfigData[15]
		if flag1 == gpuConfigData[17]
			wall_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, idx, gpuConfigData, power, limiter_flag, gamma, shared)
		elseif flag1 == gpuConfigData[18]
			interior_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, idx, gpuConfigData, power, limiter_flag, gamma, shared)
			# sync_threads()
		elseif flag1 == gpuConfigData[19]
			outer_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, idx, gpuConfigData, power, limiter_flag, gamma, shared)
			# sync_threads()
		# else
		# 	@cuprintf("Warning: There is problem with the flux flags %f \n", gpuGlobalDataFixedPoint[idx].flag_1)
		end
	else
		return nothing
	end
	# sync_threads()
	return nothing
end

@inline function wall_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, idx, gpuConfigData, power, limiter_flag, gamma, shared)
	# gpuGlobalDataRest[5, idx] = 0.0
	# gpuGlobalDataRest[6, idx] = 0.0
	# gpuGlobalDataRest[7, idx] = 0.0
	# gpuGlobalDataRest[8, idx] = 0.0
	wall_dGx_pos_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, idx, gpuConfigData, power, limiter_flag, gamma, shared)
	# sync_threads()
	wall_dGx_neg_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, idx, gpuConfigData, power, limiter_flag, gamma, shared)
	# sync_threads()
	wall_dGy_neg_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, idx, gpuConfigData, power, limiter_flag, gamma, shared)
	return nothing
end

@inline function interior_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, idx, gpuConfigData, power, limiter_flag, gamma, shared)
	# gpuGlobalDataRest[5, idx] = 0.0
	# gpuGlobalDataRest[6, idx] = 0.0
	# gpuGlobalDataRest[7, idx] = 0.0
	# gpuGlobalDataRest[8, idx] = 0.0
	interior_dGx_pos_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, idx, gpuConfigData, power, limiter_flag, gamma, shared)
	# sync_threads()
	interior_dGx_neg_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, idx, gpuConfigData, power, limiter_flag, gamma, shared)
	# sync_threads()
	interior_dGy_pos_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, idx, gpuConfigData, power, limiter_flag, gamma, shared)
	# sync_threads()
	interior_dGy_neg_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, idx, gpuConfigData, power, limiter_flag, gamma, shared)
	return nothing
end

@inline function outer_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, idx, gpuConfigData, power, limiter_flag, gamma, shared)
	# gpuGlobalDataRest[5, idx] = 0.0
	# gpuGlobalDataRest[6, idx] = 0.0
	# gpuGlobalDataRest[7, idx] = 0.0
	# gpuGlobalDataRest[8, idx] = 0.0
	outer_dGx_pos_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, idx, gpuConfigData, power, limiter_flag, gamma, shared)
	# sync_threads()
	outer_dGx_neg_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, idx, gpuConfigData, power, limiter_flag, gamma, shared)
	# sync_threads()
	outer_dGy_pos_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, idx, gpuConfigData, power, limiter_flag, gamma, shared)
	return nothing
end