function venkat_limiter(qtilde, globaldata, idx, configData)
    VL_CONST = configData["core"]["vl_const"]
    ds = globaldata[idx].short_distance
    epsi = VL_CONST * ds
    epsi = epsi ^ 3
    phi = Array{Float64,1}(undef, 0)
    del_pos = zero(Float64)
    del_neg = zero(Float64)
    for i in 1:4
        q = globaldata[idx].q[i]
        del_neg = qtilde[i] - q
        if abs(del_neg) <= 1e-5
            push!(phi, 1.0)
        elseif abs(del_neg) > 1e-5
            if del_neg > 0.0
                max_q = maximum(globaldata, idx, i)
                del_pos = max_q - q
            elseif del_neg < 0.0
                min_q = minimum(globaldata, idx, i)
                del_pos = min_q - q
            end
            num = (del_pos*del_pos) + (epsi*epsi)
            num = num*del_neg + 2 *del_neg*del_neg*del_pos

            den = del_pos*del_pos + 2 *del_neg*del_neg
            den = den + del_neg*del_pos + epsi*epsi
            den = den*del_neg

            temp = num/den

            if temp < 1.0
                push!(phi, temp)
            else
                push!(phi, 1.0)
            end
        end
    end
    return phi
end

function maximum(globaldata, idx::Int, i::Int)
    maxval::Float64 = globaldata[idx].q[i]
    for itm in globaldata[idx].conn
        if maxval < globaldata[itm].q[i]
            maxval = globaldata[itm].q[i]
        end
    end
    return maxval
end

function minimum(globaldata, idx::Int, i::Int)
    minval::Float64 = globaldata[idx].q[i]
    for itm in globaldata[idx].conn
        if minval > globaldata[itm].q[i]
            minval = globaldata[itm].q[i]
        end
    end
    return minval
end

function smallest_dist(globaldata, idx::Int)
    min_dist::Float64 = 1000.0
    ds::Float64 = zero(Float64)
    for itm in globaldata[idx].conn
        ds = hypot(globaldata[idx].x - globaldata[itm].x, globaldata[idx].y - globaldata[itm].y)
        if ds < min_dist
            min_dist = ds
        end
    end
    return min_dist
end

function max_q_values(globaldata, idx)
    maxq = globaldata[idx].q

    for itm in globaldata[idx].conn
        currq = globaldata[itm].q
        for i in 1:4
            if maxq[i] < currq[i]
                maxq[i] = currq[i]
            end
        end
    end
    return maxq
end

function min_q_values(globaldata, idx)
    minq = globaldata[idx].q
    for itm in globaldata[idx].conn
        currq = globaldata[itm].q
        for i in 1:4
            if minq[i] > currq[i]
                minq[i] = currq[i]
            end
        end
    end
    return minq
end

function qtilde_to_primitive(qtilde, configData)

    gamma = configData["core"]["gamma"]
    q1 = qtilde[1]
    q2 = qtilde[2]
    q3 = qtilde[3]
    q4 = qtilde[4]

    beta = -q4 * 0.5

    temp = 0.5/beta

    u1 = q2*temp
    u2 = q3*temp

    temp1 = q1 + beta*(u1*u1 + u2*u2)
    temp2 = temp1 - (log(beta)/(gamma-1))
    rho = exp(temp2)
    pr = rho*temp
    return (u1,u2,rho,pr)
end
