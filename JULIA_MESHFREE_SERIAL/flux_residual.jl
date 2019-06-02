function cal_flux_residual(globaldata, wallindices, outerindices, interiorindices, configData)
	phi_i = zeros(Float64,4)
    phi_k = zeros(Float64,4)
	wallindices_flux_residual(globaldata, configData, wallindices, phi_i, phi_k)
	outerindices_flux_residual(globaldata, configData, outerindices, phi_i, phi_k)
	interiorindices_flux_residual(globaldata, configData, interiorindices, phi_i, phi_k)
	return nothing
end

function wallindices_flux_residual(globaldata, configData, wallindices, phi_i, phi_k)
	for itm in wallindices
		# println(itm)
		Gxp = wall_dGx_pos(globaldata, itm, configData, phi_i, phi_k)
		Gxn = wall_dGx_neg(globaldata, itm, configData, phi_i, phi_k)
		Gyn = wall_dGy_neg(globaldata, itm, configData, phi_i, phi_k)
		GTemp = @.((Gxp + Gxn + Gyn) * 2)
		globaldata[itm].flux_res = GTemp
		# if itm == 3
		# 	println(IOContext(stdout, :compact => false), Gxp)
		# 	println(IOContext(stdout, :compact => false), Gxp + Gxn)
		# 	println(IOContext(stdout, :compact => false), Gxp + Gxn + Gyn)
		# end
	end
	return nothing
end

function outerindices_flux_residual(globaldata, configData, outerindices, phi_i, phi_k)
	for itm in outerindices
		Gxp = outer_dGx_pos(globaldata, itm, configData, phi_i, phi_k)
		Gxn = outer_dGx_neg(globaldata, itm, configData, phi_i, phi_k)
		Gyp = outer_dGy_pos(globaldata, itm, configData, phi_i, phi_k)
		GTemp = @.(Gxp + Gxn + Gyp)
		globaldata[itm].flux_res = GTemp
	end
	return nothing
end

function interiorindices_flux_residual(globaldata, configData, interiorindices, phi_i, phi_k)
	for itm in interiorindices
		Gxp = interior_dGx_pos(globaldata, itm, configData, phi_i, phi_k)
		Gxn = interior_dGx_neg(globaldata, itm, configData, phi_i, phi_k)
		Gyp = interior_dGy_pos(globaldata, itm, configData, phi_i, phi_k)
		Gyn = interior_dGy_neg(globaldata, itm, configData, phi_i, phi_k)
		# if itm == 1
		# 	println("=======")
		# 	println(IOContext(stdout, :compact => false), Gxp)
		# 	println(IOContext(stdout, :compact => false), Gxn + Gxp)
		# 	println(IOContext(stdout, :compact => false), Gxn + Gxp + Gyp)
		# 	println(IOContext(stdout, :compact => false), Gxn + Gxp + Gyp + Gyn)
		# 	println()
		# end
		GTemp = @.(Gxp + Gxn + Gyp + Gyn)
		globaldata[itm].flux_res = GTemp
	end
	return nothing
end
