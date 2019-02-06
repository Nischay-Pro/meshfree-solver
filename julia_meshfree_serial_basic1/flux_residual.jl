function cal_flux_residual(globaldata, wallindices, outerindices, interiorindices, configData)
	max_q = Vector{Float64}(undef, 1)
	min_q = Vector{Float64}(undef, 1)
	wallindices_flux_residual(globaldata, configData, wallindices, max_q, min_q)
	outerindices_flux_residual(globaldata, configData, outerindices, max_q, min_q)
	interiorindices_flux_residual(globaldata, configData, interiorindices, max_q, min_q)
end

function wallindices_flux_residual(globaldata, configData, wallindices, max_q, min_q)
	for itm in wallindices
		Gxp = wall_dGx_pos(globaldata, itm, configData, max_q, min_q)
		Gxn = wall_dGx_neg(globaldata, itm, configData, max_q, min_q)
		Gyn = wall_dGy_neg(globaldata, itm, configData, max_q, min_q)
		GTemp = Gxp + Gxn + Gyn
		GTemp = GTemp * 2
		globaldata[itm].flux_res = GTemp
	end
end

function outerindices_flux_residual(globaldata, configData, outerindices, max_q, min_q)
	for itm in outerindices
		Gxp = outer_dGx_pos(globaldata, itm, configData, max_q, min_q)
		Gxn = outer_dGx_neg(globaldata, itm, configData, max_q, min_q)
		Gyp = outer_dGy_pos(globaldata, itm, configData, max_q, min_q)
		GTemp = Gxp + Gxn + Gyp
		globaldata[itm].flux_res = GTemp
	end
end

function interiorindices_flux_residual(globaldata, configData, interiorindices, max_q, min_q)
	for itm in interiorindices
		Gxp = interior_dGx_pos(globaldata, itm, configData, max_q, min_q)
		Gxn = interior_dGx_neg(globaldata, itm, configData, max_q, min_q)
		Gyp = interior_dGy_pos(globaldata, itm, configData, max_q, min_q)
		Gyn = interior_dGy_neg(globaldata, itm, configData, max_q, min_q)
		GTemp = Gxp + Gxn + Gyp + Gyn
		globaldata[itm].flux_res = GTemp
	end
end
