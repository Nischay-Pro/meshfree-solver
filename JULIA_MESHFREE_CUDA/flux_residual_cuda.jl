function cal_flux_residual_kernel(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, gpuConfigData, numPoints)
	idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
	# thread_idx = (threadIdx().x - UInt32(1)) * UInt32(8)
	shared = @cuStaticSharedMem(Float64, 1024)
	flux_shared = @cuStaticSharedMem(Float64, 512)
	if idx > 0 && idx <= numPoints

		flux_shared[threadIdx().x], flux_shared[threadIdx().x + blockDim().x], flux_shared[threadIdx().x + blockDim().x * UInt32(2)],
			flux_shared[threadIdx().x + blockDim().x * UInt32(3)] = 0.0, 0.0, 0.0, 0.0

		gpuGlobalDataRest[21, idx], gpuGlobalDataRest[22, idx], gpuGlobalDataRest[23, idx], gpuGlobalDataRest[24, idx] = gpuGlobalDataRest[9, idx],
			gpuGlobalDataRest[10, idx],gpuGlobalDataRest[11, idx],gpuGlobalDataRest[12, idx]
		gpuGlobalDataRest[25, idx], gpuGlobalDataRest[26, idx], gpuGlobalDataRest[27, idx], gpuGlobalDataRest[28, idx] = gpuGlobalDataRest[9, idx],
			gpuGlobalDataRest[10, idx],gpuGlobalDataRest[11, idx],gpuGlobalDataRest[12, idx]

        for iter in 5:14
            conn = gpuGlobalDataConn[iter, idx]
            if conn == 0.0
                break
            end
			update_q(gpuGlobalDataRest, idx, 1, conn)
			update_q(gpuGlobalDataRest, idx, 2, conn)
			update_q(gpuGlobalDataRest, idx, 3, conn)
			update_q(gpuGlobalDataRest, idx, 4, conn)
        end


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

	gpuGlobalDataRest[5, idx] = flux_shared[threadIdx().x]
	gpuGlobalDataRest[6, idx] = flux_shared[threadIdx().x + blockDim().x]
	gpuGlobalDataRest[7, idx] = flux_shared[threadIdx().x + blockDim().x * UInt32(2)]
	gpuGlobalDataRest[8, idx] = flux_shared[threadIdx().x + blockDim().x * UInt32(3)]
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