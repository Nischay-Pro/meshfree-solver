function cal_flux_residual(globaldata, wallindices, outerindices, interiorindices, configData)
	wallindices_flux_residual(globaldata, configData, wallindices)
	outerindices_flux_residual(globaldata, configData, outerindices)
	interiorindices_flux_residual(globaldata, configData, interiorindices)
end

function wallindices_flux_residual(globaldata, configData, wallindices)
	for itm in wallindices
		Gxp = wall_dGx_pos(globaldata, itm, configData)
		Gxn = wall_dGx_neg(globaldata, itm, configData)
		Gyn = wall_dGy_neg(globaldata, itm, configData)
		GTemp = Gxp + Gxn + Gyn
		# if itm == 100
		# 	println("=======")
		# 	println(IOContext(stdout, :compact => false), Gxp)
		# 	println(IOContext(stdout, :compact => false), Gxn)
		# 	println(IOContext(stdout, :compact => false), Gyn)
		# 	println()
		# end
		GTemp = GTemp * 2.0
		globaldata[itm].flux_res = GTemp
	end
end

function outerindices_flux_residual(globaldata, configData, outerindices)
	for itm in outerindices
		Gxp = outer_dGx_pos(globaldata, itm, configData)
		Gxn = outer_dGx_neg(globaldata, itm, configData)
		Gyp = outer_dGy_pos(globaldata, itm, configData)
		GTemp = Gxp + Gxn + Gyp
		globaldata[itm].flux_res = GTemp
	end
end

function interiorindices_flux_residual(globaldata, configData, interiorindices)
	for itm in interiorindices
		Gxp = interior_dGx_pos(globaldata, itm, configData)
		Gxn = interior_dGx_neg(globaldata, itm, configData)
		Gyp = interior_dGy_pos(globaldata, itm, configData)
		Gyn = interior_dGy_neg(globaldata, itm, configData)
		GTemp = @.(Gxp + Gxn + Gyp + Gyn)
		globaldata[itm].flux_res = GTemp
	end
end
