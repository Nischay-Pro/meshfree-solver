function venkat_limiter(qtilde, globaldata, idx, configData, phi)
    # smallest_dist(globaldata, idx)
    VL_CONST = configData["core"]["vl_const"]::Float64
    ds = globaldata[idx].short_distance
    epsi = VL_CONST * ds
    epsi = epsi ^ 3
    # phi = zeros(Float64, 4)
    del_pos = zero(Float64)
    del_neg = zero(Float64)
    for i in 1:4
        q = copy(globaldata[idx].q[i])
        del_neg = qtilde[i] - q
        if abs(del_neg) <= 1e-5
            phi[i] = 1.0
        elseif abs(del_neg) > 1e-5
            if del_neg > 0.0
                # maximum(globaldata, idx, i)
                del_pos = globaldata[idx].max_q[i] - q
            elseif del_neg < 0.0
                # minimum(globaldata, idx, i)
                del_pos = globaldata[idx].min_q[i] - q
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

# @inline function maximum(globaldata, idx, i)
#     globaldata[idx].max_q[i] = copy(globaldata[idx].q[i])
#     for itm in globaldata[idx].conn
#         if globaldata[idx].max_q[i] < globaldata[itm].q[i]
#             globaldata[idx].max_q[i] = globaldata[itm].q[i]
#         end
#     end
#     return  nothing
# end

# @inline function minimum(globaldata, idx, i)
#     globaldata[idx].min_q[i] = copy(globaldata[idx].q[i])
#     for itm in globaldata[idx].conn
#         if globaldata[idx].min_q[i] > globaldata[itm].q[i]
#             globaldata[idx].min_q[i] = globaldata[itm].q[i]
#         end
#     end
#     return nothing
# end

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

@inline function max_q_values(globaldata, idx)
    maxq = copy(globaldata[idx].q)
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

@inline function min_q_values(globaldata, idx)
    minq = copy(globaldata[idx].q)
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

@inline function qtilde_to_primitive(result::Array{Float64,1}, qtilde::Array{Float64,1}, configData)
    gamma::Float64 = configData["core"]["gamma"]
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
