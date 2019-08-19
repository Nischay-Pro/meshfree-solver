function cal_flux_residual_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, gpuConfigData, numPoints)
	idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
	thread_idx = threadIdx().x
	block_dim = blockDim().x
	# thread_idx = (threadIdx().x - UInt32(1)) * UInt32(8)
	shared = @cuStaticSharedMem(Float64, 1024)
	flux_shared = @cuStaticSharedMem(Float64, 512)
	if idx > 0 && idx <= numPoints

		flux_shared[thread_idx], flux_shared[thread_idx + block_dim], flux_shared[thread_idx + block_dim * 2],
			flux_shared[thread_idx + block_dim * 3] = 0.0, 0.0, 0.0, 0.0

		if gpuGlobalDataFixedPoint[idx].flag_1 == gpuConfigData[17]
			wall_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, gpuConfigData, shared, flux_shared)
		elseif gpuGlobalDataFixedPoint[idx].flag_1 == gpuConfigData[18]
			interior_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, gpuConfigData, shared, flux_shared)
		elseif gpuGlobalDataFixedPoint[idx].flag_1 == gpuConfigData[19]
			outer_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, gpuConfigData, shared, flux_shared)
		end
	else
		return nothing
	end

	gpuGlobalDataRest[5, idx] = flux_shared[thread_idx]
	gpuGlobalDataRest[6, idx] = flux_shared[thread_idx + block_dim]
	gpuGlobalDataRest[7, idx] = flux_shared[thread_idx + block_dim * 2]
	gpuGlobalDataRest[8, idx] = flux_shared[thread_idx + block_dim * 3]
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