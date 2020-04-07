function venkat_limiter(qtilde, vl_const, globaldata, index, gamma, phi)
    ds = globaldata.short_distance[index]
    epsi = vl_const * ds
    epsi = epsi * epsi * epsi
    del_pos = zero(Float64)
    del_neg = zero(Float64)
    VLBroadcaster(globaldata.q[index], qtilde, globaldata.max_q[index], globaldata.min_q[index], phi, epsi, del_pos, del_neg)
    return nothing
end

function VLBroadcaster(q, qtilde, max_q, min_q, phi, epsi, del_pos, del_neg)
    for i in 1:4
        del_neg = qtilde[i] - q[i]
        if abs(del_neg) <= 1e-5
            phi[i] = one(Float64)
        elseif abs(del_neg) > 1e-5
            if del_neg > zero(Float64)
                del_pos = max_q[i] - q[i]
            elseif del_neg < zero(Float64)
                del_pos = min_q[i] - q[i]
            end
            num = (del_pos*del_pos) + (epsi*epsi)
            num = (num*del_neg) + 2 * (del_neg*del_neg*del_pos)

            den = (del_pos*del_pos) + (2 *del_neg*del_neg)
            den = den + (del_neg*del_pos) + (epsi*epsi)
            den = den*del_neg

            temp = num/den
            if temp < one(Float64)
                phi[i] = temp
            else
                phi[i] = one(Float64)
            end
        end
    end
    return nothing
end


@inline function qtilde_to_primitive(result, qtilde, gamma)
    beta = -qtilde[4]*0.5
    temp = 0.5/beta
    u1 = qtilde[2]*temp
    u2 = qtilde[3]*temp

    temp1 = qtilde[1] + beta*(u1*u1 + u2*u2)
    temp2 = temp1 - (log(beta)/(gamma-1))
    rho = exp(temp2)
    pr = rho*temp
    result[1] = u1
    result[2] = u2
    result[3] = rho
    result[4] = pr
    return nothing
end

@inline function connectivity_stats(x_i, y_i, nx, ny, power, conn_x, conn_y, ∑_Δx_sqr, ∑_Δy_sqr, ∑_Δx_Δy)
    x_k = conn_x
    y_k = conn_y
    
    Δx = x_k - x_i
    Δy = y_k - y_i
    
    Δs = Δx*ny - Δy*nx
    Δn = Δx*nx + Δy*ny
    
    dist = sqrt(Δs*Δs+Δn*Δn)
    weights = dist ^ power
    
    Δs_weights = Δs*weights
    Δn_weights = Δn*weights
    
    ∑_Δx_sqr += Δs*Δs_weights
    ∑_Δy_sqr += Δn*Δn_weights
    ∑_Δx_Δy += Δs*Δn_weights

    return Δx, Δy, Δs_weights, Δn_weights, ∑_Δx_sqr, ∑_Δy_sqr, ∑_Δx_Δy
end

function calculate_qtile(qtilde_i, qtilde_k, globaldata, idx, conn, Δx, Δy, vl_const, gamma, limiter_flag, phi_i, phi_k)
    update_qtildes(qtilde_i, globaldata.q[idx], globaldata.dq1[idx], globaldata.dq2[idx], Δx, Δy)
    update_qtildes(qtilde_k, globaldata.q[conn], globaldata.dq1[conn], globaldata.dq2[conn], Δx, Δy)

    if limiter_flag == 1
        venkat_limiter(qtilde_i, vl_const, globaldata, idx, gamma, phi_i)
        venkat_limiter(qtilde_k, vl_const, globaldata, conn, gamma, phi_k)
        update_qtildes(qtilde_i, globaldata.q[idx], globaldata.dq1[idx], globaldata.dq2[idx], Δx, Δy, phi_i)
        update_qtildes(qtilde_k, globaldata.q[conn], globaldata.dq1[conn], globaldata.dq2[conn], Δx, Δy, phi_k)
    end
    return nothing
end

@inline function update_qtildes(qtilde, q, dq1, dq2, Δx, Δy)
    for iter in 1:4
        qtilde[iter] = q[iter] - 0.5 * (Δx * dq1[iter] + Δy * dq2[iter])
    end
    return nothing
end

@inline function update_qtildes(qtilde, q, dq1, dq2, Δx, Δy, phi)
    for iter in 1:4
        qtilde[iter] = q[iter] - 0.5 * phi[iter] * (Δx * dq1[iter] + Δy * dq2[iter])
    end
    return nothing
end

@inline function update_delf(∑_Δx_Δf, ∑_Δy_Δf, G_k, G_i, Δs_weights, Δn_weights)
    for iter in 1:4
        intermediate_var = G_k[iter] - G_i[iter]
        ∑_Δx_Δf[iter] += intermediate_var * Δs_weights
        ∑_Δy_Δf[iter] += intermediate_var * Δn_weights
    end
    return nothing
end