function venkat_limiter(qtilde, vl_const, globaldata_point, gamma, phi)
    ds = globaldata_point.short_distance
    epsi = vl_const * ds
    epsi = epsi ^ 3
    del_pos = zero(Float64)
    del_neg = zero(Float64)
    VLBroadcaster(globaldata_point.q, qtilde, globaldata_point.max_q, globaldata_point.min_q, phi, epsi, del_pos, del_neg)
    return nothing
end

function VLBroadcaster(q, qtilde, max_q, min_q, phi, epsi, del_pos, del_neg)
    for i in 1:4
        del_neg = qtilde[i] - q[i]
        if abs(del_neg) <= 1e-5
            phi[i] = 1.0
        elseif abs(del_neg) > 1e-5
            if del_neg > 0.0
                del_pos = max_q[i] - q[i]
            elseif del_neg < 0.0
                del_pos = min_q[i] - q[i]
            end
            num = (del_pos*del_pos) + (epsi*epsi)
            num = (num*del_neg) + 2 * (del_neg*del_neg*del_pos)

            den = (del_pos*del_pos) + (2 *del_neg*del_neg)
            den = den + (del_neg*del_pos) + (epsi*epsi)
            den = den*del_neg

            temp = num/den
            if temp < 1.0
                phi[i] = temp
            else
                phi[i] = 1.0
            end
        end
    end
    return nothing
end

@inline function smallest_dist(globaldata, idx)
    min_dist = 1000.0
    for itm in globaldata[idx].conn
        ds = hypot(globaldata[idx].x - globaldata[itm].x, globaldata[idx].y - globaldata[itm].y)
        if ds < min_dist
            min_dist = ds
        end
    end
    globaldata[idx].short_distance = min_dist
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
