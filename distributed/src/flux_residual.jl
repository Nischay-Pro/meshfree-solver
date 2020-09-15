function cal_flux_residual(loc_globaldata, loc_ghost_holder, power, limiter_flag, vl_const, gamma)

    phi_i = zeros(MVector{4})
	phi_k = zeros(MVector{4})
	G_i = zeros(MVector{4})
    G_k = zeros(MVector{4})
	result = zeros(MVector{4})
	qtilde_i = zeros(MVector{4})
	qtilde_k = zeros(MVector{4})
	Gxp = zeros(MVector{4})
	Gxn = zeros(MVector{4})
	Gyp = zeros(MVector{4})
	Gyn = zeros(MVector{4})
    ∑_Δx_Δf = zeros(MVector{4})
    ∑_Δy_Δf = zeros(MVector{4})


    dist_length = length(loc_globaldata)

	for idx in 1:dist_length
		if loc_globaldata[idx].flag_1 == 0
			wallindices_flux_residual(loc_globaldata, loc_ghost_holder, dist_length, gamma, idx, Gxp, Gxn, Gyp, Gyn, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, ∑_Δx_Δf, ∑_Δy_Δf, power, limiter_flag, vl_const )
		elseif loc_globaldata[idx].flag_1 == 2
			outerindices_flux_residual(loc_globaldata, loc_ghost_holder, dist_length, gamma, idx, Gxp, Gxn, Gyp, Gyn, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, ∑_Δx_Δf, ∑_Δy_Δf, power, limiter_flag, vl_const )
		elseif loc_globaldata[idx].flag_1 == 1
			interiorindices_flux_residual(loc_globaldata, loc_ghost_holder, dist_length, gamma, idx, Gxp, Gxn, Gyp, Gyn, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, ∑_Δx_Δf, ∑_Δy_Δf, power, limiter_flag, vl_const)
		end
	end
	return nothing
end

function wallindices_flux_residual(loc_globaldata, loc_ghost_holder, dist_length, gamma, idx, Gxp, Gxn, Gyp, Gyn, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, ∑_Δx_Δf, ∑_Δy_Δf, power, limiter_flag, vl_const )
		wall_dGx_pos(loc_globaldata, loc_ghost_holder, dist_length, idx, gamma, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, ∑_Δx_Δf, ∑_Δy_Δf, power, limiter_flag, vl_const, Gxp )
		wall_dGx_neg(loc_globaldata, loc_ghost_holder, dist_length, idx, gamma, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, ∑_Δx_Δf, ∑_Δy_Δf, power, limiter_flag, vl_const, Gxn )
		wall_dGy_neg(loc_globaldata, loc_ghost_holder, dist_length, idx, gamma, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, ∑_Δx_Δf, ∑_Δy_Δf, power, limiter_flag, vl_const, Gyn )
		loc_globaldata[idx].flux_res = SVector{4}((Gxp + Gxn + Gyn) * 2)

	return nothing
end

function outerindices_flux_residual(loc_globaldata, loc_ghost_holder, dist_length, gamma, idx, Gxp, Gxn, Gyp, Gyn, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, ∑_Δx_Δf, ∑_Δy_Δf, power, limiter_flag, vl_const )
	outer_dGx_pos(loc_globaldata, loc_ghost_holder, dist_length, idx, gamma, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, ∑_Δx_Δf, ∑_Δy_Δf, power, limiter_flag, vl_const, Gxp )
	outer_dGx_neg(loc_globaldata, loc_ghost_holder, dist_length, idx, gamma, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, ∑_Δx_Δf, ∑_Δy_Δf, power, limiter_flag, vl_const, Gxn )
	outer_dGy_pos(loc_globaldata, loc_ghost_holder, dist_length, idx, gamma, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, ∑_Δx_Δf, ∑_Δy_Δf, power, limiter_flag, vl_const, Gyp )
	loc_globaldata[idx].flux_res = SVector{4}(Gxp + Gxn + Gyp)
	return nothing
end

function interiorindices_flux_residual(loc_globaldata, loc_ghost_holder, dist_length, gamma, idx, Gxp, Gxn, Gyp, Gyn, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, ∑_Δx_Δf, ∑_Δy_Δf, power, limiter_flag, vl_const)
	interior_dGx_pos(loc_globaldata, loc_ghost_holder, dist_length, idx, gamma, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, ∑_Δx_Δf, ∑_Δy_Δf, power, limiter_flag, vl_const, Gxp)
	interior_dGx_neg(loc_globaldata, loc_ghost_holder, dist_length, idx, gamma, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, ∑_Δx_Δf, ∑_Δy_Δf, power, limiter_flag, vl_const, Gxn)
	interior_dGy_pos(loc_globaldata, loc_ghost_holder, dist_length, idx, gamma, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, ∑_Δx_Δf, ∑_Δy_Δf, power, limiter_flag, vl_const, Gyp)
	interior_dGy_neg(loc_globaldata, loc_ghost_holder, dist_length, idx, gamma, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, ∑_Δx_Δf, ∑_Δy_Δf, power, limiter_flag, vl_const, Gyn)
	loc_globaldata[idx].flux_res = SVector{4}(Gxp + Gxn + Gyp + Gyn)
	return nothing
end
