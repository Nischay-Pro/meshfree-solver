function venkat_limiter_kernel_i(qtilde, gpuGlobalDataCommon, idx, gpuConfigData)
    VL_CONST = gpuConfigData[8]
    ds = gpuGlobalDataCommon[137, idx]
    # @cuprintf("Type is %s", typeof(VL_CONST))
    epsigh = VL_CONST * ds
    power3 = 3.0
    epsi = CUDAnative.pow(epsigh, power3)
    del_pos = 0.0
    del_neg = 0.0
    num = 0.0
    den = 0.0
    temp = 0.0

    for i in 1:4
        q = gpuGlobalDataCommon[38 + i, idx]
        del_neg = qtilde[i] - q
        if abs(del_neg) <= 1e-5
            if i == 1
                gpuGlobalDataCommon[146,idx] = 1.0
            elseif i == 2
                gpuGlobalDataCommon[147,idx] = 1.0
            elseif i == 3
                gpuGlobalDataCommon[148,idx] = 1.0
            else
                gpuGlobalDataCommon[149,idx] = 1.0
            end

        elseif abs(del_neg) > 1e-5
            if del_neg > 0.0
                # maximum(globaldata, idx, i, max_q)
                del_pos = gpuGlobalDataCommon[137+i, idx] - q
            elseif del_neg < 0.0
                # minimum(globaldata, idx, i, min_q)
                del_pos = gpuGlobalDataCommon[141+i, idx] - q
            end
            num = (del_pos*del_pos) + (epsi*epsi)
            num = (num*del_neg) + 2 * (del_neg*del_neg*del_pos)

            den = (del_pos*del_pos) + (2 *del_neg*del_neg)
            den = den + (del_neg*del_pos) + (epsi*epsi)
            den = den*del_neg

            temp = num/den
            if temp >= 1.0
                temp = 1.0
            end
            if i == 1
                gpuGlobalDataCommon[146,idx] = temp
            elseif i == 2
                gpuGlobalDataCommon[147,idx] = temp
            elseif i == 3
                gpuGlobalDataCommon[148,idx] = temp
            else
                gpuGlobalDataCommon[149,idx] = temp
            end
        end
    end
    return nothing
end

function venkat_limiter_kernel_k(qtilde, gpuGlobalDataCommon, idx, gpuConfigData, trueidx)
    VL_CONST = gpuConfigData[8]
    ds = gpuGlobalDataCommon[137, idx]
    # @cuprintf("Type is %s", typeof(VL_CONST))
    epsigh = VL_CONST * ds
    power3 = 3.0
    epsi = CUDAnative.pow(epsigh, power3)
    del_pos = 0.0
    del_neg = 0.0
    num = 0.0
    den = 0.0
    temp = 0.0

    for i in 1:4
        q = gpuGlobalDataCommon[38 + i, idx]
        del_neg = qtilde[i] - q
        if abs(del_neg) <= 1e-5
            if i == 1
                gpuGlobalDataCommon[150,trueidx] = 1.0
            elseif i == 2
                gpuGlobalDataCommon[151,trueidx] = 1.0
            elseif i == 3
                gpuGlobalDataCommon[152,trueidx] = 1.0
            else
                gpuGlobalDataCommon[153,trueidx] = 1.0
            end

        elseif abs(del_neg) > 1e-5
            if del_neg > 0.0
                # maximum(globaldata, idx, i, max_q)
                del_pos = gpuGlobalDataCommon[137+i, idx] - q
            elseif del_neg < 0.0
                # minimum(globaldata, idx, i, min_q)
                del_pos = gpuGlobalDataCommon[141+i, idx] - q
            end
            num = (del_pos*del_pos) + (epsi*epsi)
            num = (num*del_neg) + 2 * (del_neg*del_neg*del_pos)

            den = (del_pos*del_pos) + (2 *del_neg*del_neg)
            den = den + (del_neg*del_pos) + (epsi*epsi)
            den = den*del_neg

            temp = num/den
            if temp >= 1.0
                temp = 1.0
            end

            if i == 1
                gpuGlobalDataCommon[150,trueidx] = temp
            elseif i == 2
                gpuGlobalDataCommon[151,trueidx] = temp
            elseif i == 3
                gpuGlobalDataCommon[152,trueidx] = temp
            else
                gpuGlobalDataCommon[153,trueidx] = temp
            end
        end
    end
    return nothing
end

function max_q_values_kernel(gpuGlobalDataCommon, idx, maxq)
    maxq = (
                gpuGlobalDataCommon[39, idx],
                gpuGlobalDataCommon[40, idx],
                gpuGlobalDataCommon[41, idx],
                gpuGlobalDataCommon[42, idx]
            )
    for iter in 9:28
        conn = Int(gpuGlobalDataCommon[iter, idx])
        if conn == 0.0
            break
        end
        # currq = globaldata[itm].q
        if maxq[1] < gpuGlobalDataCommon[39, conn]
            maxq = (gpuGlobalDataCommon[39, conn], maxq[2], maxq[3], maxq[4])
        end
        if maxq[2] < gpuGlobalDataCommon[40, conn]
            maxq = (maxq[1], gpuGlobalDataCommon[40, conn], maxq[3], maxq[4])
        end
        if maxq[3] < gpuGlobalDataCommon[41, conn]
            maxq = (maxq[1], maxq[2], gpuGlobalDataCommon[41, conn], maxq[4])
        end
        if maxq[4] < gpuGlobalDataCommon[42, conn]
            maxq = (maxq[1], maxq[2], maxq[3], gpuGlobalDataCommon[42, conn])
        end
    end
    return nothing
end

function min_q_values_kernel(gpuGlobalDataCommon, idx, minq)
    minq = (
                gpuGlobalDataCommon[39, idx],
                gpuGlobalDataCommon[40, idx],
                gpuGlobalDataCommon[41, idx],
                gpuGlobalDataCommon[42, idx]
            )
    for iter in 9:28
        conn = Int(gpuGlobalDataCommon[iter, idx])
        if conn == 0.0
            break
        end
        # currq = globaldata[itm].q
        if minq[1] > gpuGlobalDataCommon[39, conn]
            minq = (gpuGlobalDataCommon[39, conn], minq[2], minq[3], minq[4])
        end
        if minq[2] > gpuGlobalDataCommon[40, conn]
            minq = (minq[1], gpuGlobalDataCommon[40, conn], minq[3], minq[4])
        end
        if minq[3] > gpuGlobalDataCommon[41, conn]
            minq = (minq[1], minq[2], gpuGlobalDataCommon[41, conn], minq[4])
        end
        if minq[4] > gpuGlobalDataCommon[42, conn]
            minq = (minq[1], minq[2], minq[3], gpuGlobalDataCommon[42, conn])
        end
    end
    return nothing
end

@inline function smallest_dist(globaldata, idx::Int64)
    min_dist = 1000.0
    for itm in globaldata[idx].conn
        ds = hypot(globaldata[idx].x - globaldata[itm].x, globaldata[idx].y - globaldata[itm].y)
        if ds < min_dist
            min_dist = ds
        end
    end
    globaldata[idx].short_distance = min_dist
end

function qtilde_to_primitive_kernel(qtilde, gpuConfigData, gpuGlobalDataCommon, idx)

    gamma = gpuConfigData[15]
    beta = -qtilde[4]*0.5
    temp = 0.5/beta

    u1 = qtilde[2]*temp
    u2 = qtilde[3]*temp

    temp1 = qtilde[1] + beta*(u1*u1 + u2*u2)
    temp2 = temp1 - (CUDAnative.log(beta)/(gamma-1))
    rho = CUDAnative.exp(temp2)
    pr = rho*temp
    gpuGlobalDataCommon[170, idx] = u1
    gpuGlobalDataCommon[171, idx] = u2
    gpuGlobalDataCommon[172, idx] = rho
    gpuGlobalDataCommon[173, idx] = pr
    return nothing
end
