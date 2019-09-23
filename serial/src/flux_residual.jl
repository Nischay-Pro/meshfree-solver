function cal_flux_residual(globaldata, wallindices, outerindices, interiorindices, configData, Gxp, Gxn, Gyp, Gyn, phi_i, phi_k, G_i, G_k,
		result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf)

	wallindices_flux_residual(globaldata, configData, wallindices, Gxp, Gxn, Gyp, Gyn, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf)
	outerindices_flux_residual(globaldata, configData, outerindices, Gxp, Gxn, Gyp, Gyn, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf)
	interiorindices_flux_residual(globaldata, configData, interiorindices, Gxp, Gxn, Gyp, Gyn, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf)
	return nothing
end

function wallindices_flux_residual(globaldata, configData, wallindices, Gxp, Gxn, Gyp, Gyn, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf)
	for itm in wallindices
		# println(itm)
		wall_dGx_pos(globaldata, itm, configData, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf, Gxp)
		wall_dGx_neg(globaldata, itm, configData, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf, Gxn)
		wall_dGy_neg(globaldata, itm, configData, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf, Gyn)
		# GTemp =
		@. globaldata[itm].flux_res = ((Gxp + Gxn + Gyn) * 2)
		# if itm == 3
		# 	println(IOContext(stdout, :compact => false), Gxp)
		# 	println(IOContext(stdout, :compact => false), Gxp + Gxn)
		# 	println(IOContext(stdout, :compact => false), Gxp + Gxn + Gyn)
		# end
	end
	return nothing
end

function outerindices_flux_residual(globaldata, configData, outerindices, Gxp, Gxn, Gyp, Gyn, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf)
	for itm in outerindices
		Gxp .= outer_dGx_pos(globaldata, itm, configData, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf)
		Gxn .= outer_dGx_neg(globaldata, itm, configData, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf)
		Gyp .= outer_dGy_pos(globaldata, itm, configData, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf)
		# GTemp =
		@. globaldata[itm].flux_res = (Gxp + Gxn + Gyp)
	end
	return nothing
end

function interiorindices_flux_residual(globaldata, configData, interiorindices, Gxp, Gxn, Gyp, Gyn, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf)
	for itm in interiorindices
		interior_dGx_pos(globaldata, itm, configData, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf, Gxp)
		interior_dGx_neg(globaldata, itm, configData, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf, Gxn)
		interior_dGy_pos(globaldata, itm, configData, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf, Gyp)
		interior_dGy_neg(globaldata, itm, configData, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf, Gyn)
		@. globaldata[itm].flux_res = (Gxp + Gxn + Gyp + Gyn)
	end
	return nothing
end
