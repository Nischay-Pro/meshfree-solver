function interior_dGx_pos(globaldata, idx, gamma, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf, power, limiter_flag, vl_const, Gxp)

   # power::Float64 = configData["core"]["power"]
   # limiter_flag::Float64 = configData["core"]["limiter_flag"]

    sum_delx_sqr = zero(Float64)
    sum_dely_sqr = zero(Float64)
    sum_delx_dely = zero(Float64)

    fill!(sum_delx_delf, zero(Float64))
    fill!(sum_dely_delf, zero(Float64))

    x_i = globaldata[idx].x
    y_i = globaldata[idx].y

    nx = globaldata[idx].nx
    ny = globaldata[idx].ny

    tx = ny
    ty = -nx

    for itm in globaldata[idx].xpos_conn
        globaldata_itm = globaldata[itm]
        x_k = globaldata_itm.x
        y_k = globaldata_itm.y

        delx = x_k - x_i
        dely = y_k - y_i

        dels = delx*tx + dely*ty
        deln = delx*nx + dely*ny

        dist = sqrt(dels*dels+deln*deln)
        weights = dist^power

        dels_weights = dels*weights
        deln_weights = deln*weights

        sum_delx_sqr = sum_delx_sqr + dels*dels_weights
        sum_dely_sqr = sum_dely_sqr + deln*deln_weights

        sum_delx_dely = sum_delx_dely + dels*deln_weights

        calculate_qtile(qtilde_i, qtilde_k, globaldata[idx], globaldata_itm, delx, dely, vl_const, gamma, limiter_flag, phi_i, phi_k)

        qtilde_to_primitive(result, qtilde_i, gamma)
        flux_Gxp(G_i, nx, ny, result[1], result[2], result[3], result[4])

        qtilde_to_primitive(result, qtilde_k, gamma)
        flux_Gxp(G_k, nx, ny, result[1], result[2], result[3], result[4])

        for i in 1:4
            sum_delx_delf[i] += (G_k[i] - G_i[i]) * dels_weights
            sum_dely_delf[i] += (G_k[i] - G_i[i]) * deln_weights
        end

    end
    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = one(Float64) / det
    for iter in 1:4
        Gxp[iter] = (sum_delx_delf[iter]*sum_dely_sqr - sum_dely_delf[iter]*sum_delx_dely)*one_by_det
    end
    return nothing
end

function interior_dGx_neg(globaldata, idx, gamma, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf, power, limiter_flag, vl_const, Gxn)

   # power::Float64 = configData["core"]["power"]
   # limiter_flag::Float64 = configData["core"]["limiter_flag"]

    sum_delx_sqr = zero(Float64)
    sum_dely_sqr = zero(Float64)
    sum_delx_dely = zero(Float64)

    fill!(sum_delx_delf, zero(Float64))
    fill!(sum_dely_delf, zero(Float64))

    x_i = globaldata[idx].x
    y_i = globaldata[idx].y

    nx = globaldata[idx].nx
    ny = globaldata[idx].ny

    tx = ny
    ty = -nx

    for itm in globaldata[idx].xneg_conn

        globaldata_itm = globaldata[itm]
        x_k = globaldata_itm.x
        y_k = globaldata_itm.y

        delx = x_k - x_i
        dely = y_k - y_i

        dels = delx*tx + dely*ty
        deln = delx*nx + dely*ny

        dist = sqrt(dels*dels+deln*deln)
        weights = dist^power

        dels_weights = dels*weights
        deln_weights = deln*weights

        sum_delx_sqr = sum_delx_sqr + dels*dels_weights
        sum_dely_sqr = sum_dely_sqr + deln*deln_weights

        sum_delx_dely = sum_delx_dely + dels*deln_weights

        calculate_qtile(qtilde_i, qtilde_k, globaldata[idx], globaldata_itm, delx, dely, vl_const, gamma, limiter_flag, phi_i, phi_k)

        qtilde_to_primitive(result, qtilde_i, gamma)
        flux_Gxn(G_i, nx, ny, result[1], result[2], result[3], result[4])

        qtilde_to_primitive(result, qtilde_k, gamma)
        flux_Gxn(G_k, nx, ny, result[1], result[2], result[3], result[4])

        for i in 1:4
            sum_delx_delf[i] += (G_k[i] - G_i[i]) * dels_weights
            sum_dely_delf[i] += (G_k[i] - G_i[i]) * deln_weights
        end
    end
    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = one(Float64) / det

    for iter in 1:4
        Gxn[iter] = (sum_delx_delf[iter] *sum_dely_sqr - sum_dely_delf[iter] *sum_delx_dely)*one_by_det
    end
    return nothing
end

function interior_dGy_pos(globaldata, idx, gamma, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf, power, limiter_flag, vl_const, Gyp)

    sum_delx_sqr = zero(Float64)
    sum_dely_sqr = zero(Float64)
    sum_delx_dely = zero(Float64)

    fill!(sum_delx_delf, zero(Float64))
    fill!(sum_dely_delf, zero(Float64))

    x_i = globaldata[idx].x
    y_i = globaldata[idx].y

    nx = globaldata[idx].nx
    ny = globaldata[idx].ny

    tx = ny
    ty = -nx

    for itm in globaldata[idx].ypos_conn

        globaldata_itm = globaldata[itm]
        x_k = globaldata_itm.x
        y_k = globaldata_itm.y

        delx = x_k - x_i
        dely = y_k - y_i

        dels = delx*tx + dely*ty
        deln = delx*nx + dely*ny

        dist = sqrt(dels*dels+deln*deln)
        weights = dist^power

        dels_weights = dels*weights
        deln_weights = deln*weights

        sum_delx_sqr = sum_delx_sqr + dels*dels_weights
        sum_dely_sqr = sum_dely_sqr + deln*deln_weights

        sum_delx_dely = sum_delx_dely + dels*deln_weights

        calculate_qtile(qtilde_i, qtilde_k, globaldata[idx], globaldata_itm, delx, dely, vl_const, gamma, limiter_flag, phi_i, phi_k)

        qtilde_to_primitive(result, qtilde_i, gamma)
        flux_Gyp(G_i,nx, ny, result[1], result[2], result[3], result[4])

        qtilde_to_primitive(result, qtilde_k, gamma)
        flux_Gyp(G_k, nx, ny, result[1], result[2], result[3], result[4])

        for i in 1:4
            sum_delx_delf[i] += (G_k[i] - G_i[i]) * dels_weights
            sum_dely_delf[i] += (G_k[i] - G_i[i]) * deln_weights
        end

    end
    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = one(Float64) / det
    for iter in 1:4
        Gyp[iter] = (sum_dely_delf[iter] *sum_delx_sqr - sum_delx_delf[iter] *sum_delx_dely)*one_by_det
    end
    return nothing
end

function interior_dGy_neg(globaldata, idx, gamma, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf, power, limiter_flag, vl_const, Gyn)

    sum_delx_sqr = zero(Float64)
    sum_dely_sqr = zero(Float64)
    sum_delx_dely = zero(Float64)

    fill!(sum_delx_delf, zero(Float64))
    fill!(sum_dely_delf, zero(Float64))

    x_i = globaldata[idx].x
    y_i = globaldata[idx].y

    nx = globaldata[idx].nx
    ny = globaldata[idx].ny

    tx = ny
    ty = -nx

    for itm in globaldata[idx].yneg_conn

        globaldata_itm = globaldata[itm]
        x_k = globaldata_itm.x
        y_k = globaldata_itm.y

        delx = x_k - x_i
        dely = y_k - y_i

        dels = delx*tx + dely*ty
        deln = delx*nx + dely*ny

        dist = sqrt(dels*dels+deln*deln)
        weights = dist^power

        dels_weights = dels*weights
        deln_weights = deln*weights

        sum_delx_sqr = sum_delx_sqr + dels*dels_weights
        sum_dely_sqr = sum_dely_sqr + deln*deln_weights

        sum_delx_dely = sum_delx_dely + dels*deln_weights

        calculate_qtile(qtilde_i, qtilde_k, globaldata[idx], globaldata_itm, delx, dely, vl_const, gamma, limiter_flag, phi_i, phi_k)

        qtilde_to_primitive(result, qtilde_i, gamma)
        flux_Gyn(G_i, nx, ny, result[1], result[2], result[3], result[4])

        qtilde_to_primitive(result, qtilde_k, gamma)
        flux_Gyn(G_k, nx, ny, result[1], result[2], result[3], result[4])

        for i in 1:4
            sum_delx_delf[i] += (G_k[i] - G_i[i]) * dels_weights
            sum_dely_delf[i] += (G_k[i] - G_i[i]) * deln_weights
        end

    end
    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = one(Float64) / det
    for iter in 1:4
        Gyn[iter] = (sum_dely_delf[iter]*sum_delx_sqr - sum_delx_delf[iter]*sum_delx_dely)*one_by_det
    end
    return nothing
end
