function cal_flux_residual_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, gpuConfigData, numPoints)
	idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
	thread_idx = (threadIdx().x - 1) * 8
	shared = @cuStaticSharedMem(Float64, 1024)
	flux_shared = @cuStaticSharedMem(Float64, 512)
	if idx > 0 && idx <= numPoints
		if idx <= numPoints
			flag1 = gpuGlobalDataFixedPoint[idx].flag_1

		for i in 1:4
			flux_shared[thread_idx + i] = 0.0
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

		if flag1 == gpuConfigData[17]
			wall_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, gpuConfigData, shared, flux_shared)
		elseif flag1 == gpuConfigData[18]
			interior_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, gpuConfigData, shared, flux_shared)
			# sync_threads()
		elseif flag1 == gpuConfigData[19]
			outer_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, gpuConfigData, shared, flux_shared)
			# sync_threads()
		# else
		# 	@cuprintf("Warning: There is problem with the flux flags %f \n", gpuGlobalDataFixedPoint[idx].flag_1)
		end
	else
		return nothing
	end

	gpuGlobalDataRest[5, idx] = flux_shared[thread_idx + 1]
	gpuGlobalDataRest[6, idx] = flux_shared[thread_idx + 2]
	gpuGlobalDataRest[7, idx] = flux_shared[thread_idx + 3]
	gpuGlobalDataRest[8, idx] = flux_shared[thread_idx + 4]
	# sync_threads()
	return nothing
end

@inline function wall_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, gpuConfigData, shared, flux_shared)
	wall_dGx_pos_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, gpuConfigData, shared, flux_shared)

	wall_dGx_neg_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, gpuConfigData, shared, flux_shared)

	wall_dGy_neg_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, gpuConfigData, shared, flux_shared)
	return nothing
end

@inline function interior_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, gpuConfigData, shared, flux_shared)

	interior_dGx_pos_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, gpuConfigData, shared, flux_shared)

	interior_dGx_neg_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, gpuConfigData, shared, flux_shared)

	interior_dGy_pos_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, gpuConfigData, shared, flux_shared)

	interior_dGy_neg_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, gpuConfigData, shared, flux_shared)
	# interior_dG_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, gpuConfigData, shared, flux_shared)
	return nothing
end

@inline function outer_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, gpuConfigData, shared, flux_shared)
	outer_dGx_pos_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, gpuConfigData, shared, flux_shared)
	outer_dGx_neg_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, gpuConfigData, shared, flux_shared)
	outer_dGy_pos_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, gpuConfigData, shared, flux_shared)
	return nothing
end