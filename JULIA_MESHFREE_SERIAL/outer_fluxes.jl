function outer_dGx_pos(globaldata, idx, configData, phi_i, phi_k)

    power::Float64 = configData["core"]["power"]
    limiter_flag::Float64 = configData["core"]["limiter_flag"]

    sum_delx_sqr = zero(Float64)
    sum_dely_sqr = zero(Float64)
    sum_delx_dely = zero(Float64)

    sum_delx_delf = zeros(Float64,4)
    sum_dely_delf = zeros(Float64,4)

    x_i = globaldata[idx].x
    y_i = globaldata[idx].y

    nx = globaldata[idx].nx
    ny = globaldata[idx].ny

    tx = ny
    ty = -nx

    G_i = zeros(Float64,4)
    G_k = zeros(Float64,4)
    result = zeros(Float64,4)


    for itm in globaldata[idx].xpos_conn

        x_k = globaldata[itm].x
        y_k = globaldata[itm].y

        delx = x_k - x_i
        dely = y_k - y_i

        dels = delx*tx + dely*ty
        deln = delx*nx + dely*ny

        dist = hypot(dels, deln)
        weights = dist ^ power

        dels_weights = dels*weights
        deln_weights = deln*weights

        sum_delx_sqr = sum_delx_sqr + dels*dels_weights
        sum_dely_sqr = sum_dely_sqr + deln*deln_weights

        sum_delx_dely = sum_delx_dely + dels*deln_weights

        qtilde_i = @. globaldata[idx].q - 0.5*(delx*globaldata[idx].dq[1] + dely*globaldata[idx].dq[2])
        qtilde_k = @. globaldata[itm].q - 0.5*(delx*globaldata[itm].dq[1] + dely*globaldata[itm].dq[2])

        if limiter_flag == 1
            venkat_limiter(qtilde_i, globaldata, idx, configData, phi_i)
            venkat_limiter(qtilde_k, globaldata, idx, configData, phi_k)
            qtilde_i = @. globaldata[idx].q - 0.5 * phi_i * (delx*globaldata[idx].dq[1] + dely*globaldata[idx].dq[2])
            qtilde_k = @. globaldata[itm].q - 0.5 * phi_k * (delx*globaldata[itm].dq[1] + dely*globaldata[itm].dq[2])
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
        flux_quad_GxIII(G_i, nx, ny, result[1], result[2], result[3], result[4])

        qtilde_to_primitive(result, qtilde_k, configData)
        flux_quad_GxIII(G_k, nx, ny, result[1], result[2], result[3], result[4])

        for i in 1:4
            sum_delx_delf[i] += (G_k[i] - G_i[i]) * dels_weights
            sum_dely_delf[i] += (G_k[i] - G_i[i]) * deln_weights
        end
    end

    det = @. sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1 / det

    return @. (sum_delx_delf*sum_dely_sqr - sum_dely_delf*sum_delx_dely)*one_by_det
end

function outer_dGx_neg(globaldata, idx, configData, phi_i, phi_k)

    power::Float64 = configData["core"]["power"]::Float64
    limiter_flag::Float64 = configData["core"]["limiter_flag"]::Float64

    sum_delx_sqr = zero(Float64)
    sum_dely_sqr = zero(Float64)
    sum_delx_dely = zero(Float64)

    sum_delx_delf = zeros(Float64, 4)
    sum_dely_delf = zeros(Float64, 4)

    x_i = globaldata[idx].x
    y_i = globaldata[idx].y

    nx = globaldata[idx].nx
    ny = globaldata[idx].ny

    tx = ny
    ty = -nx

    G_i = zeros(Float64,4)
    G_k = zeros(Float64,4)
    result = zeros(Float64,4)


    for itm in globaldata[idx].xneg_conn

        x_k = globaldata[itm].x
        y_k = globaldata[itm].y

        delx = x_k - x_i
        dely = y_k - y_i

        dels = delx*tx + dely*ty
        deln = delx*nx + dely*ny

        dist = hypot(dels, deln)
        weights = dist ^ power

        dels_weights = dels*weights
        deln_weights = deln*weights

        sum_delx_sqr = sum_delx_sqr + dels*dels_weights
        sum_dely_sqr = sum_dely_sqr + deln*deln_weights

        sum_delx_dely = sum_delx_dely + dels*deln_weights

        qtilde_i = @. globaldata[idx].q - 0.5*(delx*globaldata[idx].dq[1] + dely*globaldata[idx].dq[2])
        qtilde_k = @. globaldata[itm].q - 0.5*(delx*globaldata[itm].dq[1] + dely*globaldata[itm].dq[2])

        if limiter_flag == 1
            venkat_limiter(qtilde_i, globaldata, idx, configData, phi_i)
            venkat_limiter(qtilde_k, globaldata, idx, configData, phi_k)
            qtilde_i = @. globaldata[idx].q - 0.5 * phi_i * (delx*globaldata[idx].dq[1] + dely*globaldata[idx].dq[2])
            qtilde_k = @. globaldata[itm].q - 0.5 * phi_k * (delx*globaldata[itm].dq[1] + dely*globaldata[itm].dq[2])
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
        flux_quad_GxIV(G_i, nx, ny, result[1], result[2], result[3], result[4])

        qtilde_to_primitive(result, qtilde_k, configData)
        flux_quad_GxIV(G_k, nx, ny, result[1], result[2], result[3], result[4])

        for i in 1:4
            sum_delx_delf[i] += (G_k[i] - G_i[i]) * dels_weights
            sum_dely_delf[i] += (G_k[i] - G_i[i]) * deln_weights
        end
    end
    det = @. sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1 / det
    return @. (sum_delx_delf*sum_dely_sqr - sum_dely_delf*sum_delx_dely)*one_by_det
end

function outer_dGy_pos(globaldata, idx, configData, phi_i, phi_k)

    power::Float64 = configData["core"]["power"]::Float64
    limiter_flag::Float64 = configData["core"]["limiter_flag"]::Float64

    sum_delx_sqr = zero(Float64)
    sum_dely_sqr = zero(Float64)
    sum_delx_dely = zero(Float64)

    sum_delx_delf = zeros(Float64, 4)
    sum_dely_delf = zeros(Float64, 4)

    x_i = globaldata[idx].x
    y_i = globaldata[idx].y

    nx = globaldata[idx].nx
    ny = globaldata[idx].ny

    tx = ny
    ty = -nx

    G_i = zeros(Float64,4)
    G_k = zeros(Float64,4)
    result = zeros(Float64,4)


    for itm in globaldata[idx].ypos_conn
        x_k = globaldata[itm].x
        y_k = globaldata[itm].y

        delx = x_k - x_i
        dely = y_k - y_i

        dels = delx*tx + dely*ty
        deln = delx*nx + dely*ny

        dist = hypot(dels, deln)
        weights = dist ^ power

        dels_weights = dels*weights
        deln_weights = deln*weights

        sum_delx_sqr = sum_delx_sqr + dels*dels_weights
        sum_dely_sqr = sum_dely_sqr + deln*deln_weights

        sum_delx_dely = sum_delx_dely + dels*deln_weights

        qtilde_i = @. globaldata[idx].q - 0.5*(delx*globaldata[idx].dq[1] + dely*globaldata[idx].dq[2])
        qtilde_k = @. globaldata[itm].q - 0.5*(delx*globaldata[itm].dq[1] + dely*globaldata[itm].dq[2])

        if limiter_flag == 1
            venkat_limiter(qtilde_i, globaldata, idx, configData, phi_i)
            venkat_limiter(qtilde_k, globaldata, idx, configData, phi_k)
            qtilde_i = @. globaldata[idx].q - 0.5 * phi_i * (delx*globaldata[idx].dq[1] + dely*globaldata[idx].dq[2])
            qtilde_k = @. globaldata[itm].q - 0.5 * phi_k * (delx*globaldata[itm].dq[1] + dely*globaldata[itm].dq[2])
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
        flux_Gyp(G_i, nx, ny, result[1], result[2], result[3], result[4])

        qtilde_to_primitive(result, qtilde_k, configData)
        flux_Gyp(G_k, nx, ny, result[1], result[2], result[3], result[4])

        # print(sum_delx_delf)
        # print(" ")
        # print(G_k)
        # print(" ")
        # print(G_i * dels_weights)
        # print("\n")
        for i in 1:4
            sum_delx_delf[i] += (G_k[i] - G_i[i]) * dels_weights
            sum_dely_delf[i] += (G_k[i] - G_i[i]) * deln_weights
        end

    end
    det = @. sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely

    one_by_det = 1 / det

    return @. (sum_dely_delf*sum_delx_sqr - sum_delx_delf*sum_delx_dely)*one_by_det
end
