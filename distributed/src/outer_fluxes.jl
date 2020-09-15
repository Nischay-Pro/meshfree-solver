function outer_dGx_pos(loc_globaldata, loc_ghost_holder, dist_length, idx, gamma, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, ∑_Δx_Δf, ∑_Δy_Δf, power, limiter_flag, vl_const, Gxp)

    ∑_Δx_sqr = zero(Float64)
    ∑_Δy_sqr = zero(Float64)
    ∑_Δx_Δy = zero(Float64)

    fill!(∑_Δx_Δf, zero(Float64))
    fill!(∑_Δy_Δf, zero(Float64))

    x_i = loc_globaldata[idx].x
    y_i = loc_globaldata[idx].y

    nx = loc_globaldata[idx].nx
    ny = loc_globaldata[idx].ny

    tx = ny
    ty = -nx

    for itm in loc_globaldata[idx].xpos_conn
        if itm == zero(Float64)
            break
        end

        if itm <= dist_length
            globaldata_itm = loc_globaldata[itm]
        else
            globaldata_itm = loc_ghost_holder[1][itm]
        end
        x_k = globaldata_itm.x
        y_k = globaldata_itm.y

        delx = x_k - x_i
        dely = y_k - y_i

        dels = delx*tx + dely*ty
        deln = delx*nx + dely*ny

        dist = hypot(dels, deln)
        weights = dist ^ power

        dels_weights = dels*weights
        deln_weights = deln*weights

        ∑_Δx_sqr = ∑_Δx_sqr + dels*dels_weights
        ∑_Δy_sqr = ∑_Δy_sqr + deln*deln_weights

        ∑_Δx_Δy = ∑_Δx_Δy + dels*deln_weights

        @. qtilde_i = loc_globaldata[idx].q - 0.5*(delx*loc_globaldata[idx].dq1 + dely*loc_globaldata[idx].dq2)
        @. qtilde_k = globaldata_itm.q - 0.5*(delx*globaldata_itm.dq1 + dely*globaldata_itm.dq2)

        if limiter_flag == 1
            venkat_limiter(qtilde_i, loc_globaldata[idx], vl_const, phi_i)
            venkat_limiter(qtilde_k, globaldata_itm, vl_const, phi_k)
            @. qtilde_i = loc_globaldata[idx].q - 0.5 * phi_i * (delx*loc_globaldata[idx].dq1 + dely*loc_globaldata[idx].dq2)
            @. qtilde_k = globaldata_itm.q - 0.5 * phi_k * (delx*globaldata_itm.dq1 + dely*globaldata_itm.dq2)
        end

        qtilde_to_primitive(result, qtilde_i, gamma)
        flux_quad_GxIII(G_i, nx, ny, result[1], result[2], result[3], result[4])

        qtilde_to_primitive(result, qtilde_k, gamma)
        flux_quad_GxIII(G_k, nx, ny, result[1], result[2], result[3], result[4])

        for i in 1:4
            ∑_Δx_Δf[i] += (G_k[i] - G_i[i]) * dels_weights
            ∑_Δy_Δf[i] += (G_k[i] - G_i[i]) * deln_weights
        end
    end
    
    det = ∑_Δx_sqr*∑_Δy_sqr - ∑_Δx_Δy*∑_Δx_Δy
    one_by_det = one(Float64) / det
    
    for iter in 1:4
        Gxp[iter] = (∑_Δx_Δf[iter] *∑_Δy_sqr - ∑_Δy_Δf[iter] *∑_Δx_Δy)*one_by_det
    end
    return nothing
end

function outer_dGx_neg(loc_globaldata, loc_ghost_holder, dist_length, idx, gamma, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, ∑_Δx_Δf, ∑_Δy_Δf, power, limiter_flag, vl_const, Gxn)

    ∑_Δx_sqr = zero(Float64)
    ∑_Δy_sqr = zero(Float64)
    ∑_Δx_Δy = zero(Float64)

    fill!(∑_Δx_Δf, zero(Float64))
    fill!(∑_Δy_Δf, zero(Float64))

    x_i = loc_globaldata[idx].x
    y_i = loc_globaldata[idx].y

    nx = loc_globaldata[idx].nx
    ny = loc_globaldata[idx].ny

    tx = ny
    ty = -nx

    for itm in loc_globaldata[idx].xneg_conn
        if itm == zero(Float64)
            break
        end

        if itm <= dist_length
            globaldata_itm = loc_globaldata[itm]
        else
            globaldata_itm = loc_ghost_holder[1][itm]
        end
        x_k = globaldata_itm.x
        y_k = globaldata_itm.y

        delx = x_k - x_i
        dely = y_k - y_i

        dels = delx*tx + dely*ty
        deln = delx*nx + dely*ny

        dist = hypot(dels, deln)
        weights = dist ^ power

        dels_weights = dels*weights
        deln_weights = deln*weights

        ∑_Δx_sqr = ∑_Δx_sqr + dels*dels_weights
        ∑_Δy_sqr = ∑_Δy_sqr + deln*deln_weights

        ∑_Δx_Δy = ∑_Δx_Δy + dels*deln_weights

        @. qtilde_i = loc_globaldata[idx].q - 0.5*(delx*loc_globaldata[idx].dq1 + dely*loc_globaldata[idx].dq2)
        @. qtilde_k = globaldata_itm.q - 0.5*(delx*globaldata_itm.dq1 + dely*globaldata_itm.dq2)

        if limiter_flag == 1
            venkat_limiter(qtilde_i, loc_globaldata[idx], vl_const, phi_i)
            venkat_limiter(qtilde_k, globaldata_itm, vl_const, phi_k)
            @. qtilde_i = loc_globaldata[idx].q - 0.5 * phi_i * (delx*loc_globaldata[idx].dq1 + dely*loc_globaldata[idx].dq2)
            @. qtilde_k = globaldata_itm.q - 0.5 * phi_k * (delx*globaldata_itm.dq1 + dely*globaldata_itm.dq2)
        end

        qtilde_to_primitive(result, qtilde_i, gamma)
        flux_quad_GxIV(G_i, nx, ny, result[1], result[2], result[3], result[4])

        qtilde_to_primitive(result, qtilde_k, gamma)
        flux_quad_GxIV(G_k, nx, ny, result[1], result[2], result[3], result[4])

        for i in 1:4
            ∑_Δx_Δf[i] += (G_k[i] - G_i[i]) * dels_weights
            ∑_Δy_Δf[i] += (G_k[i] - G_i[i]) * deln_weights
        end
    end
    det = ∑_Δx_sqr*∑_Δy_sqr - ∑_Δx_Δy*∑_Δx_Δy
    one_by_det = one(Float64) / det
    for iter in 1:4
        Gxn[iter] = (∑_Δx_Δf[iter] * ∑_Δy_sqr - ∑_Δy_Δf[iter] * ∑_Δx_Δy)*one_by_det
    end
end

function outer_dGy_pos(loc_globaldata, loc_ghost_holder, dist_length, idx, gamma, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, ∑_Δx_Δf, ∑_Δy_Δf, power, limiter_flag, vl_const, Gyp)

    ∑_Δx_sqr = zero(Float64)
    ∑_Δy_sqr = zero(Float64)
    ∑_Δx_Δy = zero(Float64)

    fill!(∑_Δx_Δf, zero(Float64))
    fill!(∑_Δy_Δf, zero(Float64))

    x_i = loc_globaldata[idx].x
    y_i = loc_globaldata[idx].y

    nx = loc_globaldata[idx].nx
    ny = loc_globaldata[idx].ny

    tx = ny
    ty = -nx

    for itm in loc_globaldata[idx].ypos_conn
        if itm == zero(Float64)
            break
        end
        if itm <= dist_length
            globaldata_itm = loc_globaldata[itm]
        else
            globaldata_itm = loc_ghost_holder[1][itm]
        end
        x_k = globaldata_itm.x
        y_k = globaldata_itm.y

        delx = x_k - x_i
        dely = y_k - y_i

        dels = delx*tx + dely*ty
        deln = delx*nx + dely*ny

        dist = hypot(dels, deln)
        weights = dist ^ power

        dels_weights = dels*weights
        deln_weights = deln*weights

        ∑_Δx_sqr = ∑_Δx_sqr + dels*dels_weights
        ∑_Δy_sqr = ∑_Δy_sqr + deln*deln_weights

        ∑_Δx_Δy = ∑_Δx_Δy + dels*deln_weights

        @. qtilde_i = loc_globaldata[idx].q - 0.5*(delx*loc_globaldata[idx].dq1 + dely*loc_globaldata[idx].dq2)
        @. qtilde_k = globaldata_itm.q - 0.5*(delx*globaldata_itm.dq1 + dely*globaldata_itm.dq2)

        if limiter_flag == 1
            venkat_limiter(qtilde_i, loc_globaldata[idx], vl_const, phi_i)
            venkat_limiter(qtilde_k, globaldata_itm, vl_const, phi_k)
            @. qtilde_i = loc_globaldata[idx].q - 0.5 * phi_i * (delx*loc_globaldata[idx].dq1 + dely*loc_globaldata[idx].dq2)
            @. qtilde_k = globaldata_itm.q - 0.5 * phi_k * (delx*globaldata_itm.dq1 + dely*globaldata_itm.dq2)
        end

        qtilde_to_primitive(result, qtilde_i, gamma)
        flux_Gyp(G_i, nx, ny, result[1], result[2], result[3], result[4])

        qtilde_to_primitive(result, qtilde_k, gamma)
        flux_Gyp(G_k, nx, ny, result[1], result[2], result[3], result[4])

        for i in 1:4
            ∑_Δx_Δf[i] += (G_k[i] - G_i[i]) * dels_weights
            ∑_Δy_Δf[i] += (G_k[i] - G_i[i]) * deln_weights
        end

    end
    det = ∑_Δx_sqr*∑_Δy_sqr - ∑_Δx_Δy*∑_Δx_Δy
    one_by_det = one(Float64) / det
    for iter in 1:4
        Gyp[iter] = (∑_Δy_Δf[iter] * ∑_Δx_sqr - ∑_Δx_Δf[iter] * ∑_Δx_Δy)*one_by_det
    end
end
