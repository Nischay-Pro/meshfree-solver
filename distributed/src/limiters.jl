function venkat_limiter(qtilde, globaldata_itm, VL_CONST, phi)
    ds = globaldata_itm.short_distance
    epsi = VL_CONST * ds
    epsi = epsi ^ 3
    del_pos = zero(Float64)
    del_neg = zero(Float64)
    for i in 1:4
        q = globaldata_itm.q[i]
        del_neg = qtilde[i] - q
        if abs(del_neg) <= 1e-5
            phi[i] = 1.0
        elseif abs(del_neg) > 1e-5
            if del_neg > 0.0
                del_pos = globaldata_itm.max_q[i] - q
            elseif del_neg < 0.0
                del_pos = globaldata_itm.min_q[i] - q
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
    return  nothing
end
