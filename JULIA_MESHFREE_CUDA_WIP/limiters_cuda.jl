@inline function venkat_limiter_kernel(qtilde, gpuGlobalDataCommon, idx, gpuConfigData, phi)
    VL_CONST = gpuConfigData[8]
    ds = gpuGlobalDataCommon[137, idx]
    # @cuprintf("Type is %s", typeof(VL_CONST))
    epsigh = VL_CONST * ds
    power3 = 3.0
    epsi = CUDAnative.pow(epsigh, power3)
    del_pos = 0.0
    del_neg = 0.0

    for i in 1:4
        q = gpuGlobalDataCommon[38 + i, idx]
        del_neg = qtilde[i] - q
        if abs(del_neg) <= 1e-5
            phi[i] = 1.0
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
            if temp < 1.0
                phi[i] = temp
            else
                phi[i] = 1.0
            end
        end
    end
    return nothing
end

@inline function max_q_values_kernel(gpuGlobalDataCommon, idx, maxq)
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

@inline function min_q_values_kernel(gpuGlobalDataCommon, idx, minq)
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

@inline function qtilde_to_primitive_kernel(qtilde, gpuConfigData, result)

    gamma = gpuConfigData[15]
    q1 = qtilde[1]
    q2 = qtilde[2]
    q3 = qtilde[3]
    q4 = qtilde[4]

    beta = -q4*0.5

    temp = 0.5/beta

    u1 = q2*temp
    u2 = q3*temp

    temp1 = q1 + beta*(u1*u1 + u2*u2)
    temp2 = temp1 - (CUDAnative.log(beta)/(gamma-1))
    rho = CUDAnative.exp(temp2)
    pr = rho*temp
    result[1] = u1
    result[2] = u2
    result[3] = rho
    result[4] = pr
    return nothing
end
