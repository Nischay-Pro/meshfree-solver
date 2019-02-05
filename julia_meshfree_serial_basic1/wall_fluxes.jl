function wall_dGx_pos(globaldata, idx, configData)
    idx = convert(Int, idx)
    power = configData["core"]["power"]
    limiter_flag = configData["core"]["limiter_flag"]

    sum_delx_sqr = 0
    sum_dely_sqr = 0
    sum_delx_dely = 0

    sum_delx_delf = zeros(Float64, 4)
    sum_dely_delf = zeros(Float64, 4)

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

        # if idx == 77
        #     println("^^^^^^^^^^Sequence^^^^^^^^^^^")
        #     println("x_k ", x_k)
        #     println("y_k ", y_k)
        #     println("delx ", delx)
        #     println("dely ", dely)
        #     println("dist ", dist)
        #     println("weights ", weights)
        #     println("dels_weights ",dels_weights)
        #     println("deln_weights ", deln_weights)
        #     println("sum_delx_sqr  ", sum_delx_sqr )
        #     println("sum_dely_sqr  ", sum_dely_sqr )
        #     println("sum_delx_dely ", sum_delx_dely)
        # end

        qtilde_i = globaldata[idx].q - 0.5*(delx .* globaldata[idx].dq[1] + dely .* globaldata[idx].dq[2])
        qtilde_k = globaldata[itm].q - 0.5*(delx .* globaldata[itm].dq[1] + dely .* globaldata[itm].dq[2])

        # if idx == 76
        #     println(delx)
        #     println(dely)
        #     println(globaldata[idx].q)
        #     println(globaldata[idx].dq[1])
        #     println(globaldata[idx].dq[2])
        #     println(- 0.5*(delx .* globaldata[idx].dq[1]))
        #     println(dely .* globaldata[idx].dq[2])
        #     println("qtile_i ", qtilde_i)
        #     println("qtile_k ", qtilde_k)
        # end

        if limiter_flag == 1
            phi_i = venkat_limiter(qtilde_i, globaldata, idx, configData)
            phi_k = venkat_limiter(qtilde_k, globaldata, itm, configData)
            qtilde_i = globaldata[idx].q - 0.5 * phi_i .* (delx*globaldata[idx].dq[1] + dely*globaldata[idx].dq[2])
            qtilde_k = globaldata[itm].q - 0.5 * phi_k .* (delx*globaldata[itm].dq[1] + dely*globaldata[itm].dq[2])
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
        # if idx == 76
        #     println("qtile_i ", qtilde_i)
        #     println("qtile_k ", qtilde_k)
        # end

        result = qtilde_to_primitive(qtilde_i, configData)
        G_i = flux_quad_GxII(nx, ny, result[1], result[2], result[3], result[4])

        result = qtilde_to_primitive(qtilde_k, configData)
        G_k = flux_quad_GxII(nx, ny, result[1], result[2], result[3], result[4])

        sum_delx_delf = sum_delx_delf + (G_k - G_i) * dels_weights
        sum_dely_delf = sum_dely_delf + (G_k - G_i) * deln_weights
    end

    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1 / det
    G = (sum_delx_delf*sum_dely_sqr - sum_dely_delf*sum_delx_dely)*one_by_det
    # if idx == 77
    #     # println(" The sum_del are ",sum_delx_sqr, " ", sum_dely_sqr, " ", sum_delx_dely)
    #     # println(" The det is ",det)
    #     # println(" The onebydet is ", one_by_det)
    #     println(" G is ", G)
    # end
    return G
end

function wall_dGx_neg(globaldata, idx, configData)
    idx = convert(Int, idx)
    power = configData["core"]["power"]
    limiter_flag = configData["core"]["limiter_flag"]

    sum_delx_sqr = 0
    sum_dely_sqr = 0
    sum_delx_dely = 0

    sum_delx_delf = zeros(Float64, 4)
    sum_dely_delf = zeros(Float64, 4)

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

        qtilde_i = globaldata[idx].q - 0.5*(delx*globaldata[idx].dq[1] + dely*globaldata[idx].dq[2])
        qtilde_k = globaldata[itm].q - 0.5*(delx*globaldata[itm].dq[1] + dely*globaldata[itm].dq[2])

        if limiter_flag == 1
            phi_i = venkat_limiter(qtilde_i, globaldata, idx, configData)
            phi_k = venkat_limiter(qtilde_k, globaldata, itm, configData)
            qtilde_i = globaldata[idx].q - 0.5 * phi_i .* (delx*globaldata[idx].dq[1] + dely*globaldata[idx].dq[2])
            qtilde_k = globaldata[itm].q - 0.5 * phi_k .* (delx*globaldata[itm].dq[1] + dely*globaldata[itm].dq[2])
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
        result = qtilde_to_primitive(qtilde_i, configData)
        G_i = flux_quad_GxI(nx, ny, result[1], result[2], result[3], result[4])

        result = qtilde_to_primitive(qtilde_k, configData)
        G_k = flux_quad_GxI(nx, ny, result[1], result[2], result[3], result[4])

        sum_delx_delf = sum_delx_delf + (G_k - G_i) * dels_weights
        sum_dely_delf = sum_dely_delf + (G_k - G_i) * deln_weights
    end
    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1 / det

    G = (sum_delx_delf*sum_dely_sqr - sum_dely_delf*sum_delx_dely)*one_by_det
    return G
end

function wall_dGy_neg(globaldata, idx, configData)
    idx = convert(Int, idx)
    power = configData["core"]["power"]
    limiter_flag = configData["core"]["limiter_flag"]

    sum_delx_sqr = 0
    sum_dely_sqr = 0
    sum_delx_dely = 0

    sum_delx_delf = zeros(Float64, 4)
    sum_dely_delf = zeros(Float64, 4)

    x_i = globaldata[idx].x
    y_i = globaldata[idx].y

    nx = globaldata[idx].nx
    ny = globaldata[idx].ny

    tx = ny
    ty = -nx

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

        qtilde_i = globaldata[idx].q - 0.5*(delx*globaldata[idx].dq[1] + dely*globaldata[idx].dq[2])
        qtilde_k = globaldata[itm].q - 0.5*(delx*globaldata[itm].dq[1] + dely*globaldata[itm].dq[2])

        if limiter_flag == 1
            phi_i = venkat_limiter(qtilde_i, globaldata, idx, configData)
            phi_k = venkat_limiter(qtilde_k, globaldata, itm, configData)
            qtilde_i = globaldata[idx].q - 0.5 * phi_i .* (delx*globaldata[idx].dq[1] + dely*globaldata[idx].dq[2])
            qtilde_k = globaldata[itm].q - 0.5 * phi_k .* (delx*globaldata[itm].dq[1] + dely*globaldata[itm].dq[2])
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
        result = qtilde_to_primitive(qtilde_i, configData)
        G_i = flux_Gyn(nx, ny, result[1], result[2], result[3], result[4])

        result = qtilde_to_primitive(qtilde_k, configData)
        G_k = flux_Gyn(nx, ny, result[1], result[2], result[3], result[4])

        sum_delx_delf = sum_delx_delf + (G_k - G_i) * dels_weights
        sum_dely_delf = sum_dely_delf + (G_k - G_i) * deln_weights
    end

    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1 / det

    G = (sum_dely_delf*sum_delx_sqr - sum_delx_delf*sum_delx_dely)*one_by_det

    return G
end
