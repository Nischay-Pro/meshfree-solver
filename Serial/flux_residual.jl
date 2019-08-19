function cal_flux_residual(globaldata, wallindices, outerindices, interiorindices, configData)
	phi_i = zeros(Float64,4)
	phi_k = zeros(Float64,4)
	G_i = zeros(Float64,4)
    G_k = zeros(Float64,4)
	result = zeros(Float64,4)
	qtilde_i = zeros(Float64,4)
	qtilde_k = zeros(Float64,4)
	wallindices_flux_residual(globaldata, configData, wallindices, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k)
	outerindices_flux_residual(globaldata, configData, outerindices, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k)
	interiorindices_flux_residual(globaldata, configData, interiorindices, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k)
	return nothing
end

function wallindices_flux_residual(globaldata, configData, wallindices, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k)
	Gxp = zeros(Float64, 4)
	Gxn = zeros(Float64, 4)
	Gyn = zeros(Float64, 4)

	for itm in wallindices
		# println(itm)
		wall_dGx_pos(globaldata, itm, configData, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, Gxp)
		wall_dGx_neg(globaldata, itm, configData, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, Gxn)
		wall_dGy_neg(globaldata, itm, configData, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, Gyn)
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

function outerindices_flux_residual(globaldata, configData, outerindices, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k)
	Gxp = zeros(Float64, 4)
	Gxn = zeros(Float64, 4)
	Gyp = zeros(Float64, 4)
	for itm in outerindices
		Gxp .= outer_dGx_pos(globaldata, itm, configData, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k)
		Gxn .= outer_dGx_neg(globaldata, itm, configData, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k)
		Gyp .= outer_dGy_pos(globaldata, itm, configData, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k)
		# GTemp =
		@. globaldata[itm].flux_res = (Gxp + Gxn + Gyp)
	end
	return nothing
end

function interiorindices_flux_residual(globaldata, configData, interiorindices, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k)
	Gxp = zeros(Float64, 4)
	Gxn = zeros(Float64, 4)
	Gyp = zeros(Float64, 4)
	Gyn = zeros(Float64, 4)
	for itm in interiorindices
		interior_dGx_pos(globaldata, itm, configData, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, Gxp)
		interior_dGx_neg(globaldata, itm, configData, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, Gxn)
		interior_dGy_pos(globaldata, itm, configData, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, Gyp)
		interior_dGy_neg(globaldata, itm, configData, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, Gyn)
		@. globaldata[itm].flux_res = (Gxp + Gxn + Gyp + Gyn)
	end
	return nothing
end
