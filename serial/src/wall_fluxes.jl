function wall_dGx_pos(globaldata, idx, gamma, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, ∑_Δx_Δf, ∑_Δy_Δf, power, limiter_flag, vl_const, Gxp)
    
    ∑_Δx_sqr = zero(Float64)
    ∑_Δy_sqr = zero(Float64)
    ∑_Δx_Δy = zero(Float64)
    
    fill!(∑_Δx_Δf, zero(Float64))
    fill!(∑_Δy_Δf, zero(Float64))
    
    x_i = globaldata.x[idx]
    y_i = globaldata.y[idx]
    
    nx = globaldata.nx[idx]
    ny = globaldata.ny[idx]
    
    tx::Float64 = ny
    ty::Float64 = -nx
    
    
    for conn in globaldata.xpos_conn[idx]
        

        Δx, Δy, Δs_weights, Δn_weights, ∑_Δx_sqr, ∑_Δy_sqr, ∑_Δx_Δy = connectivity_stats(x_i, y_i, nx, ny, power, globaldata.x[conn], globaldata.y[conn], ∑_Δx_sqr, ∑_Δy_sqr, ∑_Δx_Δy)
        
        calculate_qtile(qtilde_i, qtilde_k, globaldata, idx, conn, Δx, Δy, vl_const, gamma, limiter_flag, phi_i, phi_k)
        
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
        #     # println(IOContext(stdout, :compact => false), Δs_weights)
        #     # println(IOContext(stdout, :compact => false), Δn_weights)
        #     println("****")
        # end
        # if idx == 3
        #     println(IOContext(stdout, :compact => false), itm)
        #     println(IOContext(stdout, :compact => false), result)
        #     println(IOContext(stdout, :compact => false), G_i)
        #     println(IOContext(stdout, :compact => false), G_k)
        #     # println(IOContext(stdout, :compact => false), Δs_weights)
        #     # println(IOContext(stdout, :compact => false), Δn_weights)
        # end
        for i in 1:4
            ∑_Δx_Δf[i] += (G_k[i] - G_i[i]) * Δs_weights
            ∑_Δy_Δf[i] += (G_k[i] - G_i[i]) * Δn_weights
        end
        # if idx == 3
        # println(IOContext(stdout, :compact => false), Δs_weights, Δn_weights)
        # println(IOContext(stdout, :compact => false), ∑_Δx_Δf)
        # println(IOContext(stdout, :compact => false), ∑_Δy_Δf)
        # end
    end
    
    det = ∑_Δx_sqr*∑_Δy_sqr - ∑_Δx_Δy*∑_Δx_Δy
    one_by_det = 1.0 / det
    @. Gxp = (∑_Δx_Δf*∑_Δy_sqr - ∑_Δy_Δf*∑_Δx_Δy)*one_by_det
    # if idx == 3
    #     println(IOContext(stdout, :compact => false), "===Gx===")
    #     # println(IOContext(stdout, :compact => false), ∑_Δx_sqr," ", ∑_Δy_sqr, " ", ∑_Δx_Δy)
    #     # println(IOContext(stdout, :compact => false), det)
    #     # println(IOContext(stdout, :compact => false), one_by_det)
    # println(IOContext(stdout, :compact => false), ∑_Δx_Δf)
    # println(IOContext(stdout, :compact => false), ∑_Δy_Δf)
    #     # println(IOContext(stdout, :compact => false), G)
    #     println()
    # end
    # return G
    return nothing
end

function wall_dGx_neg(globaldata, idx, gamma, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, ∑_Δx_Δf, ∑_Δy_Δf, power, limiter_flag, vl_const, Gxn)
    
    ∑_Δx_sqr = zero(Float64)
    ∑_Δy_sqr = zero(Float64)
    ∑_Δx_Δy = zero(Float64)
    
    fill!(∑_Δx_Δf, zero(Float64))
    fill!(∑_Δy_Δf, zero(Float64))
    
    x_i = globaldata.x[idx]
    y_i = globaldata.y[idx]
    
    nx = globaldata.nx[idx]
    ny = globaldata.ny[idx]
    
    tx = ny
    ty = -nx
    
    for conn in globaldata.xneg_conn[idx]
        

        Δx, Δy, Δs_weights, Δn_weights, ∑_Δx_sqr, ∑_Δy_sqr, ∑_Δx_Δy = connectivity_stats(x_i, y_i, nx, ny, power, globaldata.x[conn], globaldata.y[conn], ∑_Δx_sqr, ∑_Δy_sqr, ∑_Δx_Δy)
        
        calculate_qtile(qtilde_i, qtilde_k, globaldata, idx, conn, Δx, Δy, vl_const, gamma, limiter_flag, phi_i, phi_k)
        
        qtilde_to_primitive(result, qtilde_i, gamma)
        flux_quad_GxI(G_i, nx, ny, result[1], result[2], result[3], result[4])
        
        qtilde_to_primitive(result, qtilde_k, gamma)
        flux_quad_GxI(G_k, nx, ny, result[1], result[2], result[3], result[4])
        
        for i in 1:4
            ∑_Δx_Δf[i] += (G_k[i] - G_i[i]) * Δs_weights
            ∑_Δy_Δf[i] += (G_k[i] - G_i[i]) * Δn_weights
        end
        
    end
    det = ∑_Δx_sqr*∑_Δy_sqr - ∑_Δx_Δy*∑_Δx_Δy
    one_by_det = 1.0 / det
    @. Gxn = (∑_Δx_Δf*∑_Δy_sqr - ∑_Δy_Δf*∑_Δx_Δy)*one_by_det
    # return G
    return nothing
end

function wall_dGy_neg(globaldata, idx, gamma, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, ∑_Δx_Δf, ∑_Δy_Δf, power, limiter_flag, vl_const, Gyn)
    
    ∑_Δx_sqr = zero(Float64)
    ∑_Δy_sqr = zero(Float64)
    ∑_Δx_Δy = zero(Float64)
    
    fill!(∑_Δx_Δf, zero(Float64))
    fill!(∑_Δy_Δf, zero(Float64))
    
    x_i = globaldata.x[idx]
    y_i = globaldata.y[idx]
    
    nx = globaldata.nx[idx]
    ny = globaldata.ny[idx]
    
    tx = ny
    ty = -nx
    
    for conn in globaldata.yneg_conn[idx]
        

        Δx, Δy, Δs_weights, Δn_weights, ∑_Δx_sqr, ∑_Δy_sqr, ∑_Δx_Δy = connectivity_stats(x_i, y_i, nx, ny, power, globaldata.x[conn], globaldata.y[conn], ∑_Δx_sqr, ∑_Δy_sqr, ∑_Δx_Δy)
        
        calculate_qtile(qtilde_i, qtilde_k, globaldata, idx, conn, Δx, Δy, vl_const, gamma, limiter_flag, phi_i, phi_k)
        
        qtilde_to_primitive(result, qtilde_i, gamma)
        flux_Gyn(G_i, nx, ny, result[1], result[2], result[3], result[4])
        
        qtilde_to_primitive(result, qtilde_k, gamma)
        flux_Gyn(G_k, nx, ny, result[1], result[2], result[3], result[4])
        
        for i in 1:4
            ∑_Δx_Δf[i] += (G_k[i] - G_i[i]) * Δs_weights
            ∑_Δy_Δf[i] += (G_k[i] - G_i[i]) * Δn_weights
        end
        # if idx == 3
        #     println(IOContext(stdout, :compact => false), itm)
        #     println(IOContext(stdout, :compact => false), result)
        #     println(IOContext(stdout, :compact => false), G_i)
        #     println(IOContext(stdout, :compact => false), G_k)
        # end
    end
    
    det = ∑_Δx_sqr*∑_Δy_sqr - ∑_Δx_Δy*∑_Δx_Δy
    one_by_det = 1.0 / det
    @. Gyn = (∑_Δy_Δf*∑_Δx_sqr - ∑_Δx_Δf*∑_Δx_Δy)*one_by_det
    # if idx == 3
    #     println(IOContext(stdout, :compact => false), "===Gx===")
    #     println(IOContext(stdout, :compact => false), ∑_Δx_Δf)
    #     println(IOContext(stdout, :compact => false), ∑_Δy_Δf)
    #     # println(IOContext(stdout, :compact => false), G)
    #     # println()
    # end
    return nothing
end
