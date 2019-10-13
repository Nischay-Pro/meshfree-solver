function cal_flux_residual(globaldata, configData, Gxp, Gxn, Gyp, Gyn, phi_i, phi_k, G_i, G_k,
		result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf)

	dist_length = length(globaldata)
	for idx in 1:dist_length
		if globaldata[idx].flag_1 == 0
			wallindices_flux_residual(globaldata, configData, idx, Gxp, Gxn, Gyp, Gyn, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf)
		elseif globaldata[idx].flag_1 == 2
			outerindices_flux_residual(globaldata, configData, idx, Gxp, Gxn, Gyp, Gyn, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf)
		elseif globaldata[idx].flag_1 == 1
			interiorindices_flux_residual(globaldata, configData, idx, Gxp, Gxn, Gyp, Gyn, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf)
		end
	end
	return nothing
end

function wallindices_flux_residual(globaldata, configData, idx, Gxp, Gxn, Gyp, Gyn, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf)
	# for itm in wallindices
		# println(itm)
	wall_dGx_pos(globaldata, idx, configData, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf, Gxp)
	wall_dGx_neg(globaldata, idx, configData, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf, Gxn)
	wall_dGy_neg(globaldata, idx, configData, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf, Gyn)
		# GTemp =
	@. globaldata[idx].flux_res = (Gxp + Gxn + Gyn) * 2
		# if itm == 3
		# 	println(IOContext(stdout, :compact => false), 2 * Gxp)
		# 	println(IOContext(stdout, :compact => false), 2 * (Gxp + Gxn))
		# 	println(IOContext(stdout, :compact => false), 2 * (Gxp + Gxn + Gyn))
		# end
	# end
	return nothing
end

function outerindices_flux_residual(globaldata, configData, idx, Gxp, Gxn, Gyp, Gyn, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf)
	# for itm in outerindices
	Gxp .= outer_dGx_pos(globaldata, idx, configData, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf)
	Gxn .= outer_dGx_neg(globaldata, idx, configData, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf)
	Gyp .= outer_dGy_pos(globaldata, idx, configData, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf)
		# GTemp =
	@. globaldata[idx].flux_res = Gxp + Gxn + Gyp
	# end
	return nothing
end

function interiorindices_flux_residual(globaldata, configData, idx, Gxp, Gxn, Gyp, Gyn, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf)
	# for itm in interiorindices
	interior_dGx_pos(globaldata, idx, configData, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf, Gxp)
	interior_dGx_neg(globaldata, idx, configData, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf, Gxn)
	interior_dGy_pos(globaldata, idx, configData, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf, Gyp)
	interior_dGy_neg(globaldata, idx, configData, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf, Gyn)
	@. globaldata[idx].flux_res = Gxp + Gxn + Gyp + Gyn
	# end
	return nothing
end
