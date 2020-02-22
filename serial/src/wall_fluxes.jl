function wall_dGx_pos(globaldata, idx, gamma, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf, power, limiter_flag, vl_const, Gxp)
    
    sum_delx_sqr = zero(Float64)
    sum_dely_sqr = zero(Float64)
    sum_delx_dely = zero(Float64)
    
    fill!(sum_delx_delf, zero(Float64))
    fill!(sum_dely_delf, zero(Float64))
    
    x_i = globaldata[idx].x
    y_i = globaldata[idx].y
    
    nx = globaldata[idx].nx
    ny = globaldata[idx].ny
    
    tx::Float64 = ny
    ty::Float64 = -nx
    
    
    for itm in globaldata[idx].xpos_conn
        
        globaldata_itm = globaldata[itm]
        delx, dely, dels_weights, deln_weights, sum_delx_sqr, sum_dely_sqr, sum_delx_dely = connectivity_stats(x_i, y_i, nx, ny, power, globaldata_itm, sum_delx_sqr, sum_dely_sqr, sum_delx_dely)
        
        calculate_qtile(qtilde_i, qtilde_k, globaldata[idx], globaldata_itm, delx, dely, vl_const, gamma, limiter_flag, phi_i, phi_k)
        
        # if idx == 3
        #     println(IOContext(stdout, :compact => false), itm)
        #     println(IOContext(stdout, :compact => false), phi_i)
        #     println(IOContext(stdout, :compact => false), phi_k)
        # #     println(IOContext(stdout, :compact => false), qtilde_k)
        # end
        
        
        # if idx == 76
        #     println("qtile_i ", qtilde_i)
        #     println("qtile_k ", qtilde_k)
        # end
        
        flag = 1
        # if idx == 3
        #     println(IOContext(stdout, :compact => false), result)
        # end
        qtilde_to_primitive(result, qtilde_i, gamma)
        # if idx == 3
        #     println(IOContext(stdout, :compact => false), " => ", result)
        # end
        flux_quad_GxII(G_i, nx, ny, result[1], result[2], result[3], result[4], flag)
        # if idx == 3
        #     println(IOContext(stdout, :compact => false), result)
        # end
        qtilde_to_primitive(result, qtilde_k, gamma)
        # if idx == 3
        #     println(IOContext(stdout, :compact => false), " => ", result)
        # end
        flux_quad_GxII(G_k, nx, ny, result[1], result[2], result[3], result[4], flag)
        # if idx == 3
        #     println(IOContext(stdout, :compact => false), " => ", G_i)
        # end
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
        for i in 1:4
            sum_delx_delf[i] += (G_k[i] - G_i[i]) * dels_weights
            sum_dely_delf[i] += (G_k[i] - G_i[i]) * deln_weights
        end
        # if idx == 3
        # println(IOContext(stdout, :compact => false), dels_weights, deln_weights)
        # println(IOContext(stdout, :compact => false), sum_delx_delf)
        # println(IOContext(stdout, :compact => false), sum_dely_delf)
        # end
    end
    
    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1.0 / det
    @. Gxp = (sum_delx_delf*sum_dely_sqr - sum_dely_delf*sum_delx_dely)*one_by_det
    # if idx == 3
    #     println(IOContext(stdout, :compact => false), "===Gx===")
    #     # println(IOContext(stdout, :compact => false), sum_delx_sqr," ", sum_dely_sqr, " ", sum_delx_dely)
    #     # println(IOContext(stdout, :compact => false), det)
    #     # println(IOContext(stdout, :compact => false), one_by_det)
    # println(IOContext(stdout, :compact => false), sum_delx_delf)
    # println(IOContext(stdout, :compact => false), sum_dely_delf)
    #     # println(IOContext(stdout, :compact => false), G)
    #     println()
    # end
    # return G
    return nothing
end

function wall_dGx_neg(globaldata, idx, gamma, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf, power, limiter_flag, vl_const, Gxn)
    
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
        delx, dely, dels_weights, deln_weights, sum_delx_sqr, sum_dely_sqr, sum_delx_dely = connectivity_stats(x_i, y_i, nx, ny, power, globaldata_itm, sum_delx_sqr, sum_dely_sqr, sum_delx_dely)
        
        calculate_qtile(qtilde_i, qtilde_k, globaldata[idx], globaldata_itm, delx, dely, vl_const, gamma, limiter_flag, phi_i, phi_k)
        
        qtilde_to_primitive(result, qtilde_i, gamma)
        flux_quad_GxI(G_i, nx, ny, result[1], result[2], result[3], result[4])
        
        qtilde_to_primitive(result, qtilde_k, gamma)
        flux_quad_GxI(G_k, nx, ny, result[1], result[2], result[3], result[4])
        
        for i in 1:4
            sum_delx_delf[i] += (G_k[i] - G_i[i]) * dels_weights
            sum_dely_delf[i] += (G_k[i] - G_i[i]) * deln_weights
        end
        
    end
    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1.0 / det
    @. Gxn = (sum_delx_delf*sum_dely_sqr - sum_dely_delf*sum_delx_dely)*one_by_det
    # return G
    return nothing
end

function wall_dGy_neg(globaldata, idx, gamma, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, sum_delx_delf, sum_dely_delf, power, limiter_flag, vl_const, Gyn)
    
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
        delx, dely, dels_weights, deln_weights, sum_delx_sqr, sum_dely_sqr, sum_delx_dely = connectivity_stats(x_i, y_i, nx, ny, power, globaldata_itm, sum_delx_sqr, sum_dely_sqr, sum_delx_dely)
        
        calculate_qtile(qtilde_i, qtilde_k, globaldata[idx], globaldata_itm, delx, dely, vl_const, gamma, limiter_flag, phi_i, phi_k)
        
        qtilde_to_primitive(result, qtilde_i, gamma)
        flux_Gyn(G_i, nx, ny, result[1], result[2], result[3], result[4])
        
        qtilde_to_primitive(result, qtilde_k, gamma)
        flux_Gyn(G_k, nx, ny, result[1], result[2], result[3], result[4])
        
        for i in 1:4
            sum_delx_delf[i] += (G_k[i] - G_i[i]) * dels_weights
            sum_dely_delf[i] += (G_k[i] - G_i[i]) * deln_weights
        end
        # if idx == 3
        #     println(IOContext(stdout, :compact => false), itm)
        #     println(IOContext(stdout, :compact => false), result)
        #     println(IOContext(stdout, :compact => false), G_i)
        #     println(IOContext(stdout, :compact => false), G_k)
        # end
    end
    
    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1.0 / det
    @. Gyn = (sum_dely_delf*sum_delx_sqr - sum_delx_delf*sum_delx_dely)*one_by_det
    # if idx == 3
    #     println(IOContext(stdout, :compact => false), "===Gx===")
    #     println(IOContext(stdout, :compact => false), sum_delx_delf)
    #     println(IOContext(stdout, :compact => false), sum_dely_delf)
    #     # println(IOContext(stdout, :compact => false), G)
    #     # println()
    # end
    return nothing
end
