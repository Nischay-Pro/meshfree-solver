function interior_dGx_pos(globaldata, idx, configData, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf, Gxp)

    power::Float64 = configData["core"]["power"]
    limiter_flag::Float64 = configData["core"]["limiter_flag"]

    sum_delx_sqr = zero(Float64)
    sum_dely_sqr = zero(Float64)
    sum_delx_dely = zero(Float64)

    fill!(sum_delx_delf,0.0)
    fill!(sum_dely_delf, 0.0)

    x_i = globaldata[idx].x
    y_i = globaldata[idx].y

    nx = globaldata[idx].nx
    ny = globaldata[idx].ny

    tx = ny
    ty = -nx

    # fill!(G_i, 0)
    # fill!(G_k, 0)
    # fill!(result, 0)

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

        for i in 1:4
            qtilde_i[i] = (globaldata[idx].q[i]) - 0.5 * (delx*(globaldata[idx].dq[1][i]) + dely*(globaldata[idx].dq[2][i]))
            qtilde_k[i] = (globaldata[itm].q[i]) - 0.5 * (delx*(globaldata[itm].dq[1][i]) + dely*(globaldata[itm].dq[2][i]))
        end

        if limiter_flag == 1
            venkat_limiter(qtilde_i, globaldata, idx, configData, phi_i)
            venkat_limiter(qtilde_k, globaldata, itm, configData, phi_k)
            for i in 1:4
                qtilde_i[i] = (globaldata[idx].q[i]) - 0.5 * phi_i[i] * (delx*(globaldata[idx].dq[1][i]) + dely*(globaldata[idx].dq[2][i]))
                qtilde_k[i] = (globaldata[itm].q[i]) - 0.5 * phi_k[i] * (delx*(globaldata[itm].dq[1][i]) + dely*(globaldata[itm].dq[2][i]))
            end
            # if idx == 1
            #     print("The len is ", size(globaldata[idx].xpos_conn))
            #     print("\n *****",itm, " ", phi_i, " ",phi_k, " ", qtilde_i, " ", qtilde_k, "****")
            # end
        end

        if limiter_flag == 2
            maxi = max_q_values(globaldata, idx)
            mini = min_q_values(globaldata, idx)

            for i in 1:4
                if qtilde_i[i] > maxi[i]
                    qtilde_i[i] = maxi[i]
                end
                if qtilde_i[i] < mini[i]
                    qtilde_i[i] = mini[i]
                end
                if qtilde_k[i] > maxi[i]
                    qtilde_k[i] = maxi[i]
                end
                if qtilde_k[i] < mini[i]
                    qtilde_k[i] = mini[i]
                end
            end
        end
        qtilde_to_primitive(result, qtilde_i, configData)
        flux_Gxp(G_i, nx, ny, result[1], result[2], result[3], result[4])

        qtilde_to_primitive(result, qtilde_k, configData)
        flux_Gxp(G_k, nx, ny, result[1], result[2], result[3], result[4])

        for i in 1:4
            sum_delx_delf[i] += (G_k[i] - G_i[i]) * dels_weights
            sum_dely_delf[i] += (G_k[i] - G_i[i]) * deln_weights
        end
    end
    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1 / det
    @. Gxp = (sum_delx_delf*sum_dely_sqr - sum_dely_delf*sum_delx_dely)*one_by_det
    return nothing
end

function interior_dGx_neg(globaldata, idx, configData, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf, Gxn)

    power::Float64 = configData["core"]["power"]
    limiter_flag::Float64 = configData["core"]["limiter_flag"]

    sum_delx_sqr = zero(Float64)
    sum_dely_sqr = zero(Float64)
    sum_delx_dely = zero(Float64)

    fill!(sum_delx_delf,0.0)
    fill!(sum_dely_delf, 0.0)

    x_i = globaldata[idx].x
    y_i = globaldata[idx].y

    nx = globaldata[idx].nx
    ny = globaldata[idx].ny

    tx = ny
    ty = -nx

    # fill!(G_i, 0)
    # fill!(G_k, 0)
    # fill!(result, 0)

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

        for i in 1:4
            qtilde_i[i] = (globaldata[idx].q[i]) - 0.5 * (delx*(globaldata[idx].dq[1][i]) + dely*(globaldata[idx].dq[2][i]))
            qtilde_k[i] = (globaldata[itm].q[i]) - 0.5 * (delx*(globaldata[itm].dq[1][i]) + dely*(globaldata[itm].dq[2][i]))
        end

        if limiter_flag == 1
            venkat_limiter(qtilde_i, globaldata, idx, configData, phi_i)
            venkat_limiter(qtilde_k, globaldata, itm, configData, phi_k)
            for i in 1:4
                qtilde_i[i] = (globaldata[idx].q[i]) - 0.5 * phi_i[i] * (delx*(globaldata[idx].dq[1][i]) + dely*(globaldata[idx].dq[2][i]))
                qtilde_k[i] = (globaldata[itm].q[i]) - 0.5 * phi_k[i] * (delx*(globaldata[itm].dq[1][i]) + dely*(globaldata[itm].dq[2][i]))
            end
        end
        if limiter_flag == 2
            maxi = max_q_values(globaldata, idx)
            mini = min_q_values(globaldata, idx)

            for i in 1:4
                if qtilde_i[i] > maxi[i]
                    qtilde_i[i] = maxi[i]
                end
                if qtilde_i[i] < mini[i]
                    qtilde_i[i] = mini[i]
                end
                if qtilde_k[i] > maxi[i]
                    qtilde_k[i] = maxi[i]
                end
                if qtilde_k[i] < mini[i]
                    qtilde_k[i] = mini[i]
                end
            end
        end
        qtilde_to_primitive(result, qtilde_i, configData)
        flux_Gxn(G_i, nx, ny, result[1], result[2], result[3], result[4])

        qtilde_to_primitive(result, qtilde_k, configData)
        flux_Gxn(G_k, nx, ny, result[1], result[2], result[3], result[4])

        for i in 1:4
            sum_delx_delf[i] += (G_k[i] - G_i[i]) * dels_weights
            sum_dely_delf[i] += (G_k[i] - G_i[i]) * deln_weights
        end
    end
    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1 / det
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

function interior_dGy_pos(globaldata, idx, configData, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf, Gyp)

    power::Float64 = configData["core"]["power"]
    limiter_flag::Float64 = configData["core"]["limiter_flag"]

    sum_delx_sqr = zero(Float64)
    sum_dely_sqr = zero(Float64)
    sum_delx_dely = zero(Float64)

    fill!(sum_delx_delf,0.0)
    fill!(sum_dely_delf, 0.0)

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

        for i in 1:4
            qtilde_i[i] = (globaldata[idx].q[i]) - 0.5 * (delx*(globaldata[idx].dq[1][i]) + dely*(globaldata[idx].dq[2][i]))
            qtilde_k[i] = (globaldata[itm].q[i]) - 0.5 * (delx*(globaldata[itm].dq[1][i]) + dely*(globaldata[itm].dq[2][i]))
        end

        if limiter_flag == 1
            venkat_limiter(qtilde_i, globaldata, idx, configData, phi_i)
            venkat_limiter(qtilde_k, globaldata, itm, configData, phi_k)
            for i in 1:4
                qtilde_i[i] = (globaldata[idx].q[i]) - 0.5 * phi_i[i] * (delx*(globaldata[idx].dq[1][i]) + dely*(globaldata[idx].dq[2][i]))
                qtilde_k[i] = (globaldata[itm].q[i]) - 0.5 * phi_k[i] * (delx*(globaldata[itm].dq[1][i]) + dely*(globaldata[itm].dq[2][i]))
            end
        end
        if limiter_flag == 2
            maxi = max_q_values(globaldata, idx)
            mini = min_q_values(globaldata, idx)

            for i in 1:4
                if qtilde_i[i] > maxi[i]
                    qtilde_i[i] = maxi[i]
                end
                if qtilde_i[i] < mini[i]
                    qtilde_i[i] = mini[i]
                end
                if qtilde_k[i] > maxi[i]
                    qtilde_k[i] = maxi[i]
                end
                if qtilde_k[i] < mini[i]
                    qtilde_k[i] = mini[i]
                end
            end
        end
        qtilde_to_primitive(result, qtilde_i, configData)
        flux_Gyp(G_i,nx, ny, result[1], result[2], result[3], result[4])

        qtilde_to_primitive(result, qtilde_k, configData)
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
    one_by_det = 1 / det
    @. Gyp = (sum_dely_delf*sum_delx_sqr - sum_delx_delf*sum_delx_dely)*one_by_det
    return nothing
end

function interior_dGy_neg(globaldata, idx, configData, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf, Gyn)

    power::Float64 = configData["core"]["power"]
    limiter_flag::Float64 = configData["core"]["limiter_flag"]

    sum_delx_sqr = zero(Float64)
    sum_dely_sqr = zero(Float64)
    sum_delx_dely = zero(Float64)

    fill!(sum_delx_delf,0.0)
    fill!(sum_dely_delf, 0.0)

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

        for i in 1:4
            qtilde_i[i] = (globaldata[idx].q[i]) - 0.5 * (delx*(globaldata[idx].dq[1][i]) + dely*(globaldata[idx].dq[2][i]))
            qtilde_k[i] = (globaldata[itm].q[i]) - 0.5 * (delx*(globaldata[itm].dq[1][i]) + dely*(globaldata[itm].dq[2][i]))
        end

        if limiter_flag == 1
            venkat_limiter(qtilde_i, globaldata, idx, configData, phi_i)
            venkat_limiter(qtilde_k, globaldata, itm, configData, phi_k)
            for i in 1:4
                qtilde_i[i] = (globaldata[idx].q[i]) - 0.5 * phi_i[i] * (delx*(globaldata[idx].dq[1][i]) + dely*(globaldata[idx].dq[2][i]))
                qtilde_k[i] = (globaldata[itm].q[i]) - 0.5 * phi_k[i] * (delx*(globaldata[itm].dq[1][i]) + dely*(globaldata[itm].dq[2][i]))
            end
        end
        if limiter_flag == 2
            maxi = max_q_values(globaldata, idx)
            mini = min_q_values(globaldata, idx)

            for i in 1:4
                if qtilde_i[i] > maxi[i]
                    qtilde_i[i] = maxi[i]
                end
                if qtilde_i[i] < mini[i]
                    qtilde_i[i] = mini[i]
                end
                if qtilde_k[i] > maxi[i]
                    qtilde_k[i] = maxi[i]
                end
                if qtilde_k[i] < mini[i]
                    qtilde_k[i] = mini[i]
                end
            end
        end
        qtilde_to_primitive(result, qtilde_i, configData)
        flux_Gyn(G_i, nx, ny, result[1], result[2], result[3], result[4])

        qtilde_to_primitive(result, qtilde_k, configData)
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
    one_by_det = 1 / det
    @. Gyn = (sum_dely_delf*sum_delx_sqr - sum_delx_delf*sum_delx_dely)*one_by_det
    return nothing
end
