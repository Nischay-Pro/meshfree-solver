function cal_flux_residual(globaldata, configData, Gxp, Gxn, Gyp, Gyn, phi_i, phi_k, G_i, G_k,
        result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf)
        
    power::Float64 = configData["core"]["power"]
    limiter_flag::Float64 = configData["core"]["limiter_flag"]
    vl_const::Float64 = configData["core"]["vl_const"]
    gamma::Float64 = configData["core"]["gamma"]

    dist_length = length(globaldata)
	for idx in 1:dist_length
        if globaldata[idx].flag_1 == 0
            wallindices_flux_residual(globaldata, gamma, idx, Gxp, Gxn, Gyp, Gyn, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf, power, limiter_flag, vl_const)
        elseif globaldata[idx].flag_1 == 2
			outerindices_flux_residual(globaldata, gamma, idx, Gxp, Gxn, Gyp, Gyn, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf, power, limiter_flag, vl_const)
        elseif globaldata[idx].flag_1 == 1
			interiorindices_flux_residual(globaldata, gamma, idx, Gxp, Gxn, Gyp, Gyn, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf, power, limiter_flag, vl_const)
		end
    end
	return nothing
end

function wallindices_flux_residual(globaldata, gamma, idx, Gxp, Gxn, Gyp, Gyn, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf, power, limiter_flag, vl_const)
	# for itm in wallindices
		# println(itm)
	wall_dGx_pos(globaldata, idx, gamma, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf, power, limiter_flag, vl_const, Gxp)
	wall_dGx_neg(globaldata, idx, gamma, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf, power, limiter_flag, vl_const, Gxn)
	wall_dGy_neg(globaldata, idx, gamma, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf, power, limiter_flag, vl_const, Gyn)
	@. globaldata[idx].flux_res = (Gxp + Gxn + Gyn) * 2
		# if itm == 3
		# 	println(IOContext(stdout, :compact => false), 2 * Gxp)
		# 	println(IOContext(stdout, :compact => false), 2 * (Gxp + Gxn))
		# 	println(IOContext(stdout, :compact => false), 2 * (Gxp + Gxn + Gyn))
		# end
	# end
	return nothing
end

function outerindices_flux_residual(globaldata, gamma, idx, Gxp, Gxn, Gyp, Gyn, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf, power, limiter_flag, vl_const)
    # for itm in outerindices
    # @timeit to "flux_res1" begin
	outer_dGx_pos(globaldata, idx, gamma, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf, power, limiter_flag, vl_const, Gxp)
	outer_dGx_neg(globaldata, idx, gamma, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf, power, limiter_flag, vl_const, Gxn)
	outer_dGy_pos(globaldata, idx, gamma, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf, power, limiter_flag, vl_const, Gyn)
    @. globaldata[idx].flux_res = Gxp + Gxn + Gyp
	return nothing
end

function interiorindices_flux_residual(globaldata, gamma, idx, Gxp, Gxn, Gyp, Gyn, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf, power, limiter_flag, vl_const)
    # for itm in interiorindices
	interior_dGx_pos(globaldata, idx, gamma, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf, power, limiter_flag, vl_const, Gxp)
	interior_dGx_neg(globaldata, idx, gamma, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf, power, limiter_flag, vl_const, Gxn)
	interior_dGy_pos(globaldata, idx, gamma, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf, power, limiter_flag, vl_const, Gyp)
	interior_dGy_neg(globaldata, idx, gamma, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf, power, limiter_flag, vl_const, Gyn) 
    @. globaldata[idx].flux_res = Gxp + Gxn + Gyp + Gyn
	return nothing
end
