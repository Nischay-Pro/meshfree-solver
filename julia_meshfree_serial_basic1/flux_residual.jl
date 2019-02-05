function cal_flux_residual(globaldata, wallindices, outerindices, interiorindices, configData)
	for itm in wallindices
		Gxp = wall_dGx_pos(globaldata, itm, configData)
		Gxn = wall_dGx_neg(globaldata, itm, configData)
		Gyn = wall_dGy_neg(globaldata, itm, configData)

		GTemp = Gxp + Gxn + Gyn
		GTemp = GTemp * 2
		globaldata[itm].flux_res = GTemp
	end
	for itm in outerindices
		Gxp = outer_dGx_pos(globaldata, itm, configData)
		Gxn = outer_dGx_neg(globaldata, itm, configData)
		Gyp = outer_dGy_pos(globaldata, itm, configData)

		GTemp = Gxp + Gxn + Gyp
		globaldata[itm].flux_res = GTemp
	end
	for itm in interiorindices
		Gxp = interior_dGx_pos(globaldata, itm, configData)
		Gxn = interior_dGx_neg(globaldata, itm, configData)
		Gyp = interior_dGy_pos(globaldata, itm, configData)
		Gyn = interior_dGy_neg(globaldata, itm, configData)

		GTemp = Gxp + Gxn + Gyp + Gyn
		globaldata[itm].flux_res = GTemp
	end
	return globaldata
end
