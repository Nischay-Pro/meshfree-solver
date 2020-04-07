function cal_flux_residual(globaldata, numPoints, configData, Gxp, Gxn, Gyp, Gyn, phi_i, phi_k, G_i, G_k,
        result, qtilde_i, qtilde_k, ∑_Δx_Δf, ∑_Δy_Δf, main_store)
        
    power::Float64 = main_store[53]
    limiter_flag = main_store[55]
    vl_const = main_store[56]
    gamma = main_store[59]

	for idx in 1:numPoints
        if globaldata.flag_1[idx] == 0
            wallindices_flux_residual(globaldata, gamma, idx, Gxp, Gxn, Gyp, Gyn, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, ∑_Δx_Δf, ∑_Δy_Δf, power, limiter_flag, vl_const)
        elseif globaldata.flag_1[idx] == 2
			outerindices_flux_residual(globaldata, gamma, idx, Gxp, Gxn, Gyp, Gyn, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, ∑_Δx_Δf, ∑_Δy_Δf, power, limiter_flag, vl_const)
        elseif globaldata.flag_1[idx] == 1
			interiorindices_flux_residual(globaldata, gamma, idx, Gxp, Gxn, Gyp, Gyn, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, ∑_Δx_Δf, ∑_Δy_Δf, power, limiter_flag, vl_const)
		end
    end
	return nothing
end

function wallindices_flux_residual(globaldata, gamma, idx, Gxp, Gxn, Gyp, Gyn, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, ∑_Δx_Δf, ∑_Δy_Δf, power, limiter_flag, vl_const)
	wall_dGx_pos(globaldata, idx, gamma, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, ∑_Δx_Δf, ∑_Δy_Δf, power, limiter_flag, vl_const, Gxp)
	wall_dGx_neg(globaldata, idx, gamma, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, ∑_Δx_Δf, ∑_Δy_Δf, power, limiter_flag, vl_const, Gxn)
	wall_dGy_neg(globaldata, idx, gamma, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, ∑_Δx_Δf, ∑_Δy_Δf, power, limiter_flag, vl_const, Gyn)
    @. Gxp = (Gxp + Gxn + Gyn) * 2
    globaldata.flux_res[idx] = SVector{4}(Gxp)
	return nothing
end

function outerindices_flux_residual(globaldata, gamma, idx, Gxp, Gxn, Gyp, Gyn, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, ∑_Δx_Δf, ∑_Δy_Δf, power, limiter_flag, vl_const)
    # for itm in outerindices
    # @timeit to "flux_res1" begin
	outer_dGx_pos(globaldata, idx, gamma, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, ∑_Δx_Δf, ∑_Δy_Δf, power, limiter_flag, vl_const, Gxp)
	outer_dGx_neg(globaldata, idx, gamma, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, ∑_Δx_Δf, ∑_Δy_Δf, power, limiter_flag, vl_const, Gxn)
	outer_dGy_pos(globaldata, idx, gamma, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, ∑_Δx_Δf, ∑_Δy_Δf, power, limiter_flag, vl_const, Gyp)
    @. Gxp = Gxp + Gxn + Gyp
    globaldata.flux_res[idx] = SVector{4}(Gxp)
	return nothing
end

function interiorindices_flux_residual(globaldata, gamma, idx, Gxp, Gxn, Gyp, Gyn, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, ∑_Δx_Δf, ∑_Δy_Δf, power, limiter_flag, vl_const)
    # for itm in interiorindices
    # @timeit to "flux_res_gxp" begin
	    interior_dGx_pos(globaldata, idx, gamma, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, ∑_Δx_Δf, ∑_Δy_Δf, power, limiter_flag, vl_const, Gxp)
    # end
    interior_dGx_neg(globaldata, idx, gamma, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, ∑_Δx_Δf, ∑_Δy_Δf, power, limiter_flag, vl_const, Gxn)
	interior_dGy_pos(globaldata, idx, gamma, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, ∑_Δx_Δf, ∑_Δy_Δf, power, limiter_flag, vl_const, Gyp)
	interior_dGy_neg(globaldata, idx, gamma, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, ∑_Δx_Δf, ∑_Δy_Δf, power, limiter_flag, vl_const, Gyn) 
    @. Gxp = Gxp + Gxn + Gyp + Gyn
    globaldata.flux_res[idx] = SVector{4}(Gxp)
	return nothing
end
