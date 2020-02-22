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
        x_k = globaldata[itm].x
        y_k = globaldata[itm].y

        delx = x_k - x_i
        dely = y_k - y_i

        dels = delx*tx + dely*ty
        deln = delx*nx + dely*ny

        dist = hypot(dels, deln)
        weights = dist^power

        dels_weights = dels*weights
        deln_weights = deln*weights

        sum_delx_sqr = sum_delx_sqr + dels*dels_weights
        sum_dely_sqr = sum_dely_sqr + deln*deln_weights

        sum_delx_dely = sum_delx_dely + dels*deln_weights

        @. qtilde_i = globaldata[idx].q - 0.5*(delx * globaldata[idx].dq[1] + dely * globaldata[idx].dq[2])
        @. qtilde_k = globaldata[itm].q - 0.5*(delx * globaldata[itm].dq[1] + dely * globaldata[itm].dq[2])

        if limiter_flag == 1
            venkat_limiter(qtilde_i, vl_const, globaldata[idx], gamma, phi_i)
            venkat_limiter(qtilde_k, vl_const, globaldata[itm], gamma, phi_k)
            @. qtilde_i = globaldata[idx].q - 0.5 * phi_i * (delx*globaldata[idx].dq[1] + dely*globaldata[idx].dq[2])
            @. qtilde_k = globaldata[itm].q - 0.5 * phi_k * (delx*globaldata[itm].dq[1] + dely*globaldata[itm].dq[2])
            # if idx == 1
            #     print("The len is ", size(globaldata[idx].xpos_conn))
            #     print("\n *****",itm, " ", phi_i, " ",phi_k, " ", qtilde_i, " ", qtilde_k, "****")
            # end
        end


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
    @. Gxp = (sum_delx_delf*sum_dely_sqr - sum_dely_delf*sum_delx_dely)*one_by_det
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

        x_k = globaldata[itm].x
        y_k = globaldata[itm].y

        delx = x_k - x_i
        dely = y_k - y_i

        dels = delx*tx + dely*ty
        deln = delx*nx + dely*ny

        dist = hypot(dels, deln)
        weights = dist^power

        dels_weights = dels*weights
        deln_weights = deln*weights

        sum_delx_sqr = sum_delx_sqr + dels*dels_weights
        sum_dely_sqr = sum_dely_sqr + deln*deln_weights

        sum_delx_dely = sum_delx_dely + dels*deln_weights

        @. qtilde_i = globaldata[idx].q - 0.5*(delx * globaldata[idx].dq[1] + dely * globaldata[idx].dq[2])
        @. qtilde_k = globaldata[itm].q - 0.5*(delx * globaldata[itm].dq[1] + dely * globaldata[itm].dq[2])

        if limiter_flag == 1
            venkat_limiter(qtilde_i, vl_const, globaldata[idx], gamma, phi_i)
            venkat_limiter(qtilde_k, vl_const, globaldata[itm], gamma, phi_k)
            @. qtilde_i = globaldata[idx].q - 0.5 * phi_i * (delx*globaldata[idx].dq[1] + dely*globaldata[idx].dq[2])
            @. qtilde_k = globaldata[itm].q - 0.5 * phi_k * (delx*globaldata[itm].dq[1] + dely*globaldata[itm].dq[2])
        end

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
    # if idx == 1
    #     println(IOContext(stdout, :compact => false), "===Gyn===")
    #     println(IOContext(stdout, :compact => false), sum_delx_sqr)
    #     println(IOContext(stdout, :compact => false), sum_dely_sqr)
    #     println(IOContext(stdout, :compact => false), sum_delx_dely)
    #     println(IOContext(stdout, :compact => false), det)
    #     # println(IOContext(stdout, :compact => false), one_by_det)
    #     println(IOContext(stdout, :compact => false), sum_delx_delf)
    #     println(IOContext(stdout, :compact => false), sum_dely_delf)
    #     # println(IOContext(stdout, :compact => false), G)
    #     println()
    # end
    @. Gxn = (sum_delx_delf*sum_dely_sqr - sum_dely_delf*sum_delx_dely)*one_by_det
    return nothing
end

function interior_dGy_pos(globaldata, idx, gamma, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf, power, limiter_flag, vl_const, Gyp)

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

    # fill!(G_i, 0)
    # fill!(G_k, 0)
    # fill!(result, 0)

    for itm in globaldata[idx].ypos_conn

        x_k = globaldata[itm].x
        y_k = globaldata[itm].y

        delx = x_k - x_i
        dely = y_k - y_i

        dels = delx*tx + dely*ty
        deln = delx*nx + dely*ny

        dist = hypot(dels, deln)
        weights = dist^power

        dels_weights = dels*weights
        deln_weights = deln*weights

        sum_delx_sqr = sum_delx_sqr + dels*dels_weights
        sum_dely_sqr = sum_dely_sqr + deln*deln_weights

        sum_delx_dely = sum_delx_dely + dels*deln_weights

        @. qtilde_i = globaldata[idx].q - 0.5*(delx * globaldata[idx].dq[1] + dely * globaldata[idx].dq[2])
        @. qtilde_k = globaldata[itm].q - 0.5*(delx * globaldata[itm].dq[1] + dely * globaldata[itm].dq[2])

        if limiter_flag == 1
            venkat_limiter(qtilde_i, vl_const, globaldata[idx], gamma, phi_i)
            venkat_limiter(qtilde_k, vl_const, globaldata[itm], gamma, phi_k)
            @. qtilde_i = globaldata[idx].q - 0.5 * phi_i * (delx*globaldata[idx].dq[1] + dely*globaldata[idx].dq[2])
            @. qtilde_k = globaldata[itm].q - 0.5 * phi_k * (delx*globaldata[itm].dq[1] + dely*globaldata[itm].dq[2])
        end

        qtilde_to_primitive(result, qtilde_i, gamma)
        flux_Gyp(G_i,nx, ny, result[1], result[2], result[3], result[4])

        qtilde_to_primitive(result, qtilde_k, gamma)
        flux_Gyp(G_k, nx, ny, result[1], result[2], result[3], result[4])

        for i in 1:4
            sum_delx_delf[i] += (G_k[i] - G_i[i]) * dels_weights
            sum_dely_delf[i] += (G_k[i] - G_i[i]) * deln_weights
        end
        # if idx == 200
        #     println(IOContext(stdout, :compact => false), itm)
        #     println(IOContext(stdout, :compact => false), result)
        #     println(IOContext(stdout, :compact => false), G_i)
        #     println(IOContext(stdout, :compact => false), G_k)
        # end
    end
    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = one(Float64) / det
    @. Gyp = (sum_dely_delf*sum_delx_sqr - sum_delx_delf*sum_delx_dely)*one_by_det
    return nothing
end

function interior_dGy_neg(globaldata, idx, gamma, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf, power, limiter_flag, vl_const, Gyn)

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

    # fill!(G_i, 0)
    # fill!(G_k, 0)
    # fill!(result, 0)

    for itm in globaldata[idx].yneg_conn

        x_k = globaldata[itm].x
        y_k = globaldata[itm].y

        delx = x_k - x_i
        dely = y_k - y_i

        dels = delx*tx + dely*ty
        deln = delx*nx + dely*ny

        dist = hypot(dels, deln)
        weights = dist^power

        dels_weights = dels*weights
        deln_weights = deln*weights

        sum_delx_sqr = sum_delx_sqr + dels*dels_weights
        sum_dely_sqr = sum_dely_sqr + deln*deln_weights

        sum_delx_dely = sum_delx_dely + dels*deln_weights

        @. qtilde_i = globaldata[idx].q - 0.5*(delx * globaldata[idx].dq[1] + dely * globaldata[idx].dq[2])
        @. qtilde_k = globaldata[itm].q - 0.5*(delx * globaldata[itm].dq[1] + dely * globaldata[itm].dq[2])

        if limiter_flag == 1
            venkat_limiter(qtilde_i, vl_const, globaldata[idx], gamma, phi_i)
            venkat_limiter(qtilde_k, vl_const, globaldata[itm], gamma, phi_k)
            @. qtilde_i = globaldata[idx].q - 0.5 * phi_i * (delx*globaldata[idx].dq[1] + dely*globaldata[idx].dq[2])
            @. qtilde_k = globaldata[itm].q - 0.5 * phi_k * (delx*globaldata[itm].dq[1] + dely*globaldata[itm].dq[2])
        end

        qtilde_to_primitive(result, qtilde_i, gamma)
        flux_Gyn(G_i, nx, ny, result[1], result[2], result[3], result[4])

        qtilde_to_primitive(result, qtilde_k, gamma)
        flux_Gyn(G_k, nx, ny, result[1], result[2], result[3], result[4])

        for i in 1:4
            sum_delx_delf[i] += (G_k[i] - G_i[i]) * dels_weights
            sum_dely_delf[i] += (G_k[i] - G_i[i]) * deln_weights
        end
        # if idx == 1
        #     println(IOContext(stdout, :compact => false), itm)
        #     println(IOContext(stdout, :compact => false), result)
        #     println(IOContext(stdout, :compact => false), G_i)
        #     println(IOContext(stdout, :compact => false), G_k)
        # end
    end
    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = one(Float64) / det
    @. Gyn = (sum_dely_delf*sum_delx_sqr - sum_delx_delf*sum_delx_dely)*one_by_det
    return nothing
end
