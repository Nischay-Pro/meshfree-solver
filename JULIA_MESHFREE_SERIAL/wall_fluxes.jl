function wall_dGx_pos(globaldata, idx, configData)

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

    tx::Float64 = ny
    ty::Float64 = -nx

    G_i = zeros(Float64,4)
    G_k = zeros(Float64,4)

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

        qtilde_i = globaldata[idx].q - 0.5*(delx * globaldata[idx].dq[1] + dely * globaldata[idx].dq[2])
        qtilde_k = globaldata[itm].q - 0.5*(delx * globaldata[itm].dq[1] + dely * globaldata[itm].dq[2])

        # if idx == 3
        #     println(IOContext(stdout, :compact => false), itm)
        #     println(IOContext(stdout, :compact => false), qtilde_k)
        # end

        if limiter_flag == 1
            phi_i = venkat_limiter(qtilde_i, globaldata, idx, configData)
            phi_k = venkat_limiter(qtilde_k, globaldata, itm, configData)
            qtilde_i = globaldata[idx].q - 0.5 * phi_i .* (delx*globaldata[idx].dq[1] + dely*globaldata[idx].dq[2])
            qtilde_k = globaldata[itm].q - 0.5 * phi_k .* (delx*globaldata[itm].dq[1] + dely*globaldata[itm].dq[2])
        end

        # if idx == 3
        #     println(IOContext(stdout, :compact => false), itm)
        #     println(IOContext(stdout, :compact => false), phi_i)
        #     println(IOContext(stdout, :compact => false), phi_k)
        #     println(IOContext(stdout, :compact => false), qtilde_k)
        # end

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
        if idx == 100
            flag = 0
        else
            flag = 1
        end

        # if idx == 3
        #     println(IOContext(stdout, :compact => false), result)
        # end
        result = qtilde_to_primitive(qtilde_i, configData)
        # if idx == 3
        #     println(IOContext(stdout, :compact => false), itm)
        #     println(IOContext(stdout, :compact => false), result)
        # end
        flux_quad_GxII(G_i, nx, ny, result[1], result[2], result[3], result[4], flag)
        # if idx == 3
        #     println(IOContext(stdout, :compact => false), result)
        # end
        result = qtilde_to_primitive(qtilde_k, configData)
        # if idx == 3
        #     println(IOContext(stdout, :compact => false), result)
        # end
        flux_quad_GxII(G_k, nx, ny, result[1], result[2], result[3], result[4], flag)
        # if idx == 100
        #     println("****")
        #     println(IOContext(stdout, :compact => false), phi_i)
        #     println(IOContext(stdout, :compact => false), phi_k)
        #     println(IOContext(stdout, :compact => false), qtilde_i)
        #     println(IOContext(stdout, :compact => false), qtilde_k)
        #     println(IOContext(stdout, :compact => false), itm)
        #     println(IOContext(stdout, :compact => false), result)
        #     println(IOContext(stdout, :compact => false), G_i)
        #     println(IOContext(stdout, :compact => false), G_k)
        #     # println(IOContext(stdout, :compact => false), dels_weights)
        #     # println(IOContext(stdout, :compact => false), deln_weights)
        #     println("****")
        # end
        # if idx == 3
        #     println(IOContext(stdout, :compact => false), itm)
        #     println(IOContext(stdout, :compact => false), result)
        #     println(IOContext(stdout, :compact => false), G_i)
        #     println(IOContext(stdout, :compact => false), G_k)
        #     # println(IOContext(stdout, :compact => false), dels_weights)
        #     # println(IOContext(stdout, :compact => false), deln_weights)
        # end
        sum_delx_delf = sum_delx_delf + (G_k - G_i) * dels_weights
        sum_dely_delf = sum_dely_delf + (G_k - G_i) * deln_weights
    end

    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1.0 / det
    G = (sum_delx_delf*sum_dely_sqr - sum_dely_delf*sum_delx_dely)*one_by_det
    # if idx == 3
    #     println(IOContext(stdout, :compact => false), "===Gx===")
    #     # println(IOContext(stdout, :compact => false), sum_delx_sqr)
    #     # println(IOContext(stdout, :compact => false), sum_dely_sqr)
    #     # println(IOContext(stdout, :compact => false), sum_delx_dely)
    #     # println(IOContext(stdout, :compact => false), det)
    #     # println(IOContext(stdout, :compact => false), one_by_det)
    #     println(IOContext(stdout, :compact => false), sum_delx_delf)
    #     println(IOContext(stdout, :compact => false), sum_dely_delf)
    #     println(IOContext(stdout, :compact => false), G)
    #     println()
    # end
    return G
end

function wall_dGx_neg(globaldata, idx, configData)
    idx = convert(Int, idx)
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
        flux_quad_GxI(G_i, nx, ny, result[1], result[2], result[3], result[4])

        result = qtilde_to_primitive(qtilde_k, configData)
        flux_quad_GxI(G_k, nx, ny, result[1], result[2], result[3], result[4])

        sum_delx_delf = sum_delx_delf + (G_k - G_i) * dels_weights
        sum_dely_delf = sum_dely_delf + (G_k - G_i) * deln_weights

    end
    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1.0 / det
    G = (sum_delx_delf*sum_dely_sqr - sum_dely_delf*sum_delx_dely)*one_by_det

    return G
end

function wall_dGy_neg(globaldata, idx, configData)
    idx = convert(Int, idx)
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
        flux_Gyn(G_i, nx, ny, result[1], result[2], result[3], result[4])

        result = qtilde_to_primitive(qtilde_k, configData)
        flux_Gyn(G_k, nx, ny, result[1], result[2], result[3], result[4])

        sum_delx_delf = sum_delx_delf + (G_k - G_i) * dels_weights
        sum_dely_delf = sum_dely_delf + (G_k - G_i) * deln_weights
        # if idx == 3
        #     println(IOContext(stdout, :compact => false), itm)
        #     println(IOContext(stdout, :compact => false), result)
        #     println(IOContext(stdout, :compact => false), G_i)
        #     println(IOContext(stdout, :compact => false), G_k)
        # end
    end

    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1.0 / det
    G = (sum_dely_delf*sum_delx_sqr - sum_delx_delf*sum_delx_dely)*one_by_det
    # if idx == 3
    #     println(IOContext(stdout, :compact => false), "===Gx===")
    #     println(IOContext(stdout, :compact => false), sum_delx_delf)
    #     println(IOContext(stdout, :compact => false), sum_dely_delf)
    #     # println(IOContext(stdout, :compact => false), G)
    #     # println()
    # end
    return G
end
