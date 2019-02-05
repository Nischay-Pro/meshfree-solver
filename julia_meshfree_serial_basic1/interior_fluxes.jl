
function interior_dGx_pos(globaldata, idx, configData)

    power = configData["core"]["power"]
    limiter_flag = configData["core"]["limiter_flag"]

    sum_delx_sqr = 0
    sum_dely_sqr = 0
    sum_delx_dely = 0

    sum_delx_delf = zeros(Float64,4)
    sum_dely_delf = zeros(Float64,4)

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

        qtilde_i = (globaldata[idx].q) - 0.5*(delx*(globaldata[idx].dq[1]) + dely*(globaldata[idx].dq[2]))
        qtilde_k = (globaldata[itm].q) - 0.5*(delx*(globaldata[itm].dq[1]) + dely*(globaldata[itm].dq[2]))

        if limiter_flag == 1
            phi_i = venkat_limiter(qtilde_i, globaldata, idx, configData)
            phi_k = venkat_limiter(qtilde_k, globaldata, itm, configData)
            qtilde_i = (globaldata[idx].q) - 0.5 * phi_i .* (delx*(globaldata[idx].dq[1]) + dely*(globaldata[idx].dq[2]))
            qtilde_k = (globaldata[itm].q) - 0.5 * phi_k .* (delx*(globaldata[itm].dq[1]) + dely*(globaldata[itm].dq[2]))
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
        result = qtilde_to_primitive(qtilde_i, configData)
        G_i = flux_Gxp(nx, ny, result[1], result[2], result[3], result[4])

        result = qtilde_to_primitive(qtilde_k, configData)
        G_k = flux_Gxp(nx, ny, result[1], result[2], result[3], result[4])

        sum_delx_delf = sum_delx_delf + ((G_k) - (G_i)) * dels_weights
        sum_dely_delf = sum_dely_delf + ((G_k) - (G_i)) * deln_weights
    end
    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1 / det

    G = (sum_delx_delf*sum_dely_sqr - sum_dely_delf*sum_delx_dely)*one_by_det

    return G
end

function interior_dGx_neg(globaldata, idx, configData)

    power = configData["core"]["power"]
    limiter_flag = configData["core"]["limiter_flag"]

    sum_delx_sqr = 0
    sum_dely_sqr = 0
    sum_delx_dely = 0

    sum_delx_delf = zeros(Float64,4)
    sum_dely_delf = zeros(Float64,4)

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

        qtilde_i = (globaldata[idx].q) - 0.5*(delx*(globaldata[idx].dq[1]) + dely*(globaldata[idx].dq[2]))
        qtilde_k = (globaldata[itm].q) - 0.5*(delx*(globaldata[itm].dq[1]) + dely*(globaldata[itm].dq[2]))

        if limiter_flag == 1
            phi_i = (venkat_limiter(qtilde_i, globaldata, idx, configData))
            phi_k = (venkat_limiter(qtilde_k, globaldata, itm, configData))
            qtilde_i = (globaldata[idx].q) - 0.5 * phi_i .* (delx*(globaldata[idx].dq[1]) + dely*(globaldata[idx].dq[2]))
            qtilde_k = (globaldata[itm].q) - 0.5 * phi_k .* (delx*(globaldata[itm].dq[1]) + dely*(globaldata[itm].dq[2]))
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
        G_i = flux_Gxn(nx, ny, result[1], result[2], result[3], result[4])

        result = qtilde_to_primitive(qtilde_k, configData)
        G_k = flux_Gxn(nx, ny, result[1], result[2], result[3], result[4])

        sum_delx_delf = sum_delx_delf + ((G_k ) - (G_i )) * dels_weights
        sum_dely_delf = sum_dely_delf + ((G_k ) - (G_i )) * deln_weights
    end
    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1 / det

    G = (sum_delx_delf*sum_dely_sqr - sum_dely_delf*sum_delx_dely)*one_by_det

    return G
end

function interior_dGy_pos(globaldata, idx, configData)

    power = configData["core"]["power"]
    limiter_flag = configData["core"]["limiter_flag"]

    sum_delx_sqr = 0
    sum_dely_sqr = 0
    sum_delx_dely = 0

    sum_delx_delf = zeros(Float64,4)
    sum_dely_delf = zeros(Float64,4)

    x_i = globaldata[idx].x
    y_i = globaldata[idx].y

    nx = globaldata[idx].nx
    ny = globaldata[idx].ny

    tx = ny
    ty = -nx

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

        qtilde_i = (globaldata[idx].q) - 0.5*(delx*(globaldata[idx].dq[1]) + dely*(globaldata[idx].dq[2]))
        qtilde_k = (globaldata[itm].q) - 0.5*(delx*(globaldata[itm].dq[1]) + dely*(globaldata[itm].dq[2]))

        if limiter_flag == 1
            phi_i = (venkat_limiter(qtilde_i, globaldata, idx, configData))
            phi_k = (venkat_limiter(qtilde_k, globaldata, itm, configData))
            qtilde_i = (globaldata[idx].q) - 0.5 * phi_i .* (delx*(globaldata[idx].dq[1]) + dely*(globaldata[idx].dq[2]))
            qtilde_k = (globaldata[itm].q) - 0.5 * phi_k .* (delx*(globaldata[itm].dq[1]) + dely*(globaldata[itm].dq[2]))
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
        G_i = flux_Gyp(nx, ny, result[1], result[2], result[3], result[4])

        result = qtilde_to_primitive(qtilde_k, configData)
        G_k = flux_Gyp(nx, ny, result[1], result[2], result[3], result[4])

        sum_delx_delf = sum_delx_delf + ((G_k ) - (G_i )) * dels_weights
        sum_dely_delf = sum_dely_delf + ((G_k ) - (G_i )) * deln_weights
    end
    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1 / det
    G = (sum_dely_delf*sum_delx_sqr - sum_delx_delf*sum_delx_dely)*one_by_det
    return G
end

function interior_dGy_neg(globaldata, idx, configData)

    power = configData["core"]["power"]
    limiter_flag = configData["core"]["limiter_flag"]

    sum_delx_sqr = 0
    sum_dely_sqr = 0
    sum_delx_dely = 0

    sum_delx_delf = zeros(Float64,4)
    sum_dely_delf = zeros(Float64,4)

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

        qtilde_i = (globaldata[idx].q) - 0.5*(delx*(globaldata[idx].dq[1]) + dely*(globaldata[idx].dq[2]))
        qtilde_k = (globaldata[itm].q) - 0.5*(delx*(globaldata[itm].dq[1]) + dely*(globaldata[itm].dq[2]))

        if limiter_flag == 1
            phi_i = (venkat_limiter(qtilde_i, globaldata, idx, configData))
            phi_k = (venkat_limiter(qtilde_k, globaldata, itm, configData))
            qtilde_i = (globaldata[idx].q) - 0.5 * phi_i .* (delx*(globaldata[idx].dq[1]) + dely*(globaldata[idx].dq[2]))
            qtilde_k = (globaldata[itm].q) - 0.5 * phi_k .* (delx*(globaldata[itm].dq[1]) + dely*(globaldata[itm].dq[2]))
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

        sum_delx_delf = sum_delx_delf + ((G_k) - (G_i)) * dels_weights
        sum_dely_delf = sum_dely_delf + ((G_k) - (G_i)) * deln_weights
    end
    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1 / det

    G = (sum_dely_delf*sum_delx_sqr - sum_delx_delf*sum_delx_dely)*one_by_det
    return G
end
