function cal_flux_residual_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataFauxFixed, gpuGlobalDataRest, gpuConfigData, numPoints)
	idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
	thread_idx = threadIdx().x
	block_dim = blockDim().x
	# thread_idx = (threadIdx().x - UInt32(1)) * UInt32(8)
	# temp_sharedmem = @cuStaticSharedMem(Float64, 128)
	shared = @cuStaticSharedMem(Float64, 512)
	flux_shared = @cuStaticSharedMem(Float64, 256)
	qtilde_shared = @cuStaticSharedMem(Float64, 256)
	if idx > 0 && idx <= numPoints

		flux_shared[thread_idx], flux_shared[thread_idx + block_dim], flux_shared[thread_idx + block_dim * 2],
			flux_shared[thread_idx + block_dim * 3] = 0, 0, 0, 0

		if gpuGlobalDataFixedPoint[idx].flag_1 == gpuConfigData[17]
			wall_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataFauxFixed, gpuGlobalDataRest, gpuConfigData, numPoints, shared, flux_shared, qtilde_shared)
		elseif gpuGlobalDataFixedPoint[idx].flag_1 == gpuConfigData[18]
			interior_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataFauxFixed, gpuGlobalDataRest, gpuConfigData, numPoints, shared, flux_shared, qtilde_shared)
		elseif gpuGlobalDataFixedPoint[idx].flag_1 == gpuConfigData[19]
			outer_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataFauxFixed, gpuGlobalDataRest, gpuConfigData, numPoints, shared, flux_shared, qtilde_shared)
		end
		gpuGlobalDataRest[idx, 5] = flux_shared[thread_idx]
		gpuGlobalDataRest[idx, 6] = flux_shared[thread_idx + block_dim]
		gpuGlobalDataRest[idx, 7] = flux_shared[thread_idx + block_dim * 2]
		gpuGlobalDataRest[idx, 8] = flux_shared[thread_idx + block_dim * 3]
	else
		return nothing
	end
	# sync_threads()
	return nothing
end

function wall_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataFauxFixed, gpuGlobalDataRest, gpuConfigData, numPoints, shared, flux_shared, qtilde_shared)
	wall_dGx_pos_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataFauxFixed, gpuGlobalDataRest, gpuConfigData, numPoints, shared, flux_shared, qtilde_shared)

	wall_dGx_neg_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataFauxFixed, gpuGlobalDataRest, gpuConfigData, numPoints, shared, flux_shared, qtilde_shared)

	wall_dGy_neg_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataFauxFixed, gpuGlobalDataRest, gpuConfigData, numPoints, shared, flux_shared, qtilde_shared)
	return nothing
end

function interior_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataFauxFixed, gpuGlobalDataRest, gpuConfigData, numPoints, shared, flux_shared, qtilde_shared)

	interior_dGx_pos_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataFauxFixed, gpuGlobalDataRest, gpuConfigData, numPoints, shared, flux_shared, qtilde_shared)

	interior_dGx_neg_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataFauxFixed, gpuGlobalDataRest, gpuConfigData, numPoints, shared, flux_shared, qtilde_shared)

	interior_dGy_pos_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataFauxFixed, gpuGlobalDataRest, gpuConfigData, numPoints, shared, flux_shared, qtilde_shared)

	interior_dGy_neg_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataFauxFixed, gpuGlobalDataRest, gpuConfigData, numPoints, shared, flux_shared, qtilde_shared)
	return nothing
end

function outer_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataFauxFixed, gpuGlobalDataRest, gpuConfigData, numPoints, shared, flux_shared, qtilde_shared)
	outer_dGx_pos_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataFauxFixed, gpuGlobalDataRest, gpuConfigData, numPoints, shared, flux_shared, qtilde_shared)
	outer_dGx_neg_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataFauxFixed, gpuGlobalDataRest, gpuConfigData, numPoints, shared, flux_shared, qtilde_shared)
	outer_dGy_pos_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataFauxFixed, gpuGlobalDataRest, gpuConfigData, numPoints, shared, flux_shared, qtilde_shared)
	return nothing
end