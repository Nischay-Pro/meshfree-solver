function cal_flux_residual(globaldata, wallindices, outerindices, interiorindices, configData)
	wallindices_flux_residual(globaldata, configData, wallindices)
	outerindices_flux_residual(globaldata, configData, outerindices)
	interiorindices_flux_residual(globaldata, configData, interiorindices)
	return nothing
end

function wallindices_flux_residual(globaldata, configData, wallindices)
	for itm in wallindices
		# println(itm)
		Gxp = wall_dGx_pos(globaldata, itm, configData)
		Gxn = wall_dGx_neg(globaldata, itm, configData)
		Gyn = wall_dGy_neg(globaldata, itm, configData)
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

function outerindices_flux_residual(globaldata, configData, outerindices)
	for itm in outerindices
		Gxp = outer_dGx_pos(globaldata, itm, configData)
		Gxn = outer_dGx_neg(globaldata, itm, configData)
		Gyp = outer_dGy_pos(globaldata, itm, configData)
		GTemp = @.(Gxp + Gxn + Gyp)
		globaldata[itm].flux_res = GTemp
	end
	return nothing
end

function interiorindices_flux_residual(globaldata, configData, interiorindices)
	for itm in interiorindices
		Gxp = interior_dGx_pos(globaldata, itm, configData)
		Gxn = interior_dGx_neg(globaldata, itm, configData)
		Gyp = interior_dGy_pos(globaldata, itm, configData)
		Gyn = interior_dGy_neg(globaldata, itm, configData)
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
