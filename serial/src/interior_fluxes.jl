function interior_dGx_pos(globaldata, idx, gamma, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, ∑_Δx_Δf, ∑_Δy_Δf, power, limiter_flag, vl_const, Gxp)
    
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
    
    for conn in globaldata.xpos_conn[idx]

        Δx, Δy, Δs_weights, Δn_weights, ∑_Δx_sqr, ∑_Δy_sqr, ∑_Δx_Δy = connectivity_stats(x_i, y_i, nx, ny, power, globaldata.x[conn], globaldata.y[conn], ∑_Δx_sqr, ∑_Δy_sqr, ∑_Δx_Δy)
        
        calculate_qtile(qtilde_i, qtilde_k, globaldata, idx, conn, Δx, Δy, vl_const, gamma, limiter_flag, phi_i, phi_k)
        
        qtilde_to_primitive(result, qtilde_i, gamma)
        flux_Gxp(G_i, nx, ny, result[1], result[2], result[3], result[4])
        
        qtilde_to_primitive(result, qtilde_k, gamma)
        flux_Gxp(G_k, nx, ny, result[1], result[2], result[3], result[4])
        
        for i in 1:4
            ∑_Δx_Δf[i] += (G_k[i] - G_i[i]) * Δs_weights
            ∑_Δy_Δf[i] += (G_k[i] - G_i[i]) * Δn_weights
        end
        
    end
    det = ∑_Δx_sqr*∑_Δy_sqr - ∑_Δx_Δy*∑_Δx_Δy
    one_by_det = one(Float64) / det
    for iter in 1:4
        Gxp[iter] = (∑_Δx_Δf[iter]*∑_Δy_sqr - ∑_Δy_Δf[iter]*∑_Δx_Δy)*one_by_det
    end
    return nothing
end

function interior_dGx_neg(globaldata, idx, gamma, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, ∑_Δx_Δf, ∑_Δy_Δf, power, limiter_flag, vl_const, Gxn)
    
    # power::Float64 = configData["core"]["power"]
    # limiter_flag::Float64 = configData["core"]["limiter_flag"]
    
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
        flux_Gxn(G_i, nx, ny, result[1], result[2], result[3], result[4])
        
        qtilde_to_primitive(result, qtilde_k, gamma)
        flux_Gxn(G_k, nx, ny, result[1], result[2], result[3], result[4])
        
        for i in 1:4
            ∑_Δx_Δf[i] += (G_k[i] - G_i[i]) * Δs_weights
            ∑_Δy_Δf[i] += (G_k[i] - G_i[i]) * Δn_weights
        end
    end
    det = ∑_Δx_sqr*∑_Δy_sqr - ∑_Δx_Δy*∑_Δx_Δy
    one_by_det = one(Float64) / det
    
    for iter in 1:4
        Gxn[iter] = (∑_Δx_Δf[iter] *∑_Δy_sqr - ∑_Δy_Δf[iter] *∑_Δx_Δy)*one_by_det
    end
    return nothing
end

function interior_dGy_pos(globaldata, idx, gamma, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, ∑_Δx_Δf, ∑_Δy_Δf, power, limiter_flag, vl_const, Gyp)
    
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
    
    for conn in globaldata.ypos_conn[idx]
        

        Δx, Δy, Δs_weights, Δn_weights, ∑_Δx_sqr, ∑_Δy_sqr, ∑_Δx_Δy = connectivity_stats(x_i, y_i, nx, ny, power, globaldata.x[conn], globaldata.y[conn], ∑_Δx_sqr, ∑_Δy_sqr, ∑_Δx_Δy)
        
        calculate_qtile(qtilde_i, qtilde_k, globaldata, idx, conn, Δx, Δy, vl_const, gamma, limiter_flag, phi_i, phi_k)
        
        qtilde_to_primitive(result, qtilde_i, gamma)
        flux_Gyp(G_i,nx, ny, result[1], result[2], result[3], result[4])
        
        qtilde_to_primitive(result, qtilde_k, gamma)
        flux_Gyp(G_k, nx, ny, result[1], result[2], result[3], result[4])
        
        for i in 1:4
            ∑_Δx_Δf[i] += (G_k[i] - G_i[i]) * Δs_weights
            ∑_Δy_Δf[i] += (G_k[i] - G_i[i]) * Δn_weights
        end
        
    end
    det = ∑_Δx_sqr*∑_Δy_sqr - ∑_Δx_Δy*∑_Δx_Δy
    one_by_det = one(Float64) / det
    for iter in 1:4
        Gyp[iter] = (∑_Δy_Δf[iter] *∑_Δx_sqr - ∑_Δx_Δf[iter] *∑_Δx_Δy)*one_by_det
    end
    return nothing
end

function interior_dGy_neg(globaldata, idx, gamma, phi_i, phi_k, G_i, G_k, result, qtilde_i, qtilde_k, ∑_Δx_Δf, ∑_Δy_Δf, power, limiter_flag, vl_const, Gyn)
    
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
        
    end
    det = ∑_Δx_sqr*∑_Δy_sqr - ∑_Δx_Δy*∑_Δx_Δy
    one_by_det = one(Float64) / det
    for iter in 1:4
        Gyn[iter] = (∑_Δy_Δf[iter]*∑_Δx_sqr - ∑_Δx_Δf[iter]*∑_Δx_Δy)*one_by_det
    end
    return nothing
end
