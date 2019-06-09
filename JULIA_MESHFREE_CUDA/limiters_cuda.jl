function venkat_limiter_kernel_i(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, idx, gpuConfigData, delx, dely)
    VL_CONST = gpuConfigData[8]
    # ds = 1000.0
    # for iter in 5:14
    #     itm = gpuGlobalDataConn[iter, idx]
    #     if itm == 0.0
    #         break
    #     end
    #     min_dist = CUDAnative.hypot(gpuGlobalDataFixedPoint[idx].x - gpuGlobalDataFixedPoint[itm].x, gpuGlobalDataFixedPoint[idx].y - gpuGlobalDataFixedPoint[itm].y)
    #     if min_dist < ds
    #         ds = min_dist
    #     end
    # end
    ds = gpuGlobalDataFixedPoint[idx].short_distance
    # @cuprintf("Type is %s", typeof(VL_CONST))
    epsigh = VL_CONST * ds
    power3 = 3.0
    epsi = CUDAnative.pow(epsigh, power3)
    del_pos = 0.0
    del_neg = 0.0

    for i in 1:4
        q = gpuGlobalDataRest[8+i, idx]
        del_neg = gpuGlobalDataRest[8+i, idx] - 0.5*(delx * gpuGlobalDataRest[12+i, idx] + dely * gpuGlobalDataRest[16+i, idx]) - q
        if abs(del_neg) <= 1e-5
            gpuGlobalDataRest[28+i,idx] = 1.0
        elseif abs(del_neg) > 1e-5
            if del_neg > 0.0
                # maximum(globaldata, idx, i, max_q)
                del_pos = gpuGlobalDataRest[20+i, idx] - q
            elseif del_neg < 0.0
                # minimum(globaldata, idx, i, min_q)
                del_pos = gpuGlobalDataRest[24+i, idx] - q
            end
            num = (del_pos*del_pos) + (epsi*epsi)
            num = (num*del_neg) + 2 * (del_neg*del_neg*del_pos)

            den = (del_pos*del_pos) + (2 *del_neg*del_neg)
            den = den + (del_neg*del_pos) + (epsi*epsi)
            den = den*del_neg

            temp = num/den
            if temp > 1.0
                temp = 1.0
            end
            gpuGlobalDataRest[28+i,idx] = temp
        end
    end
    return nothing
end

function venkat_limiter_kernel_k(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, idx, gpuConfigData, trueidx, delx, dely)
    VL_CONST = gpuConfigData[8]
    # ds = 1000.0
    # for iter in 5:14
    #     itm = gpuGlobalDataConn[iter, idx]
    #     if itm == 0.0
    #         break
    #     end
    #     min_dist = CUDAnative.hypot(gpuGlobalDataFixedPoint[idx].x - gpuGlobalDataFixedPoint[itm].x, gpuGlobalDataFixedPoint[idx].y - gpuGlobalDataFixedPoint[itm].y)
    #     if min_dist < ds
    #         ds = min_dist
    #     end
    # end
    ds = gpuGlobalDataFixedPoint[idx].short_distance
    # @cuprintf("Type is %s", typeof(VL_CONST))
    epsigh = VL_CONST * ds
    power3 = 3.0
    epsi = CUDAnative.pow(epsigh, power3)
    del_pos = 0.0
    del_neg = 0.0

    for i in 1:4
        q = gpuGlobalDataRest[8+i, idx]
        del_neg = gpuGlobalDataRest[8+i, idx] - 0.5*(delx * gpuGlobalDataRest[12+i, idx] + dely * gpuGlobalDataRest[16+i, idx]) - q
        if abs(del_neg) <= 1e-5
            gpuGlobalDataRest[32+i,trueidx] = 1.0
        elseif abs(del_neg) > 1e-5
            if del_neg > 0.0
                # maximum(globaldata, idx, i, max_q)
                del_pos = gpuGlobalDataRest[20+i, idx] - q
            elseif del_neg < 0.0
                # minimum(globaldata, idx, i, min_q)
                del_pos = gpuGlobalDataRest[24+i, idx] - q
            end
            num = (del_pos*del_pos) + (epsi*epsi)
            num = (num*del_neg) + 2 * (del_neg*del_neg*del_pos)

            den = (del_pos*del_pos) + (2 *del_neg*del_neg)
            den = den + (del_neg*del_pos) + (epsi*epsi)
            den = den*del_neg

            temp = num/den
            if temp > 1.0
                temp = 1.0
            end
            gpuGlobalDataRest[32+i,trueidx] = temp
        end
    end
    return nothing
end

# function max_q_values_kernel(gpuGlobalDataCommon, idx, maxq)
#     maxq = (
#                 gpuGlobalDataRest[9, idx],
#                 gpuGlobalDataRest[10, idx],
#                 gpuGlobalDataRest[11, idx],
#                 gpuGlobalDataRest[12, idx]
#             )
#     for iter in 9:28
#         conn = Int(gpuGlobalDataCommon[iter, idx])
#         if conn == 0.0
#             break
#         end
#         # currq = globaldata[itm].q
#         if maxq[1] < gpuGlobalDataRest[9, conn]
#             maxq = (gpuGlobalDataRest[9, conn], maxq[2], maxq[3], maxq[4])
#         end
#         if maxq[2] < gpuGlobalDataRest[10, conn]
#             maxq = (maxq[1], gpuGlobalDataRest[10, conn], maxq[3], maxq[4])
#         end
#         if maxq[3] < gpuGlobalDataRest[11, conn]
#             maxq = (maxq[1], maxq[2], gpuGlobalDataRest[11, conn], maxq[4])
#         end
#         if maxq[4] < gpuGlobalDataRest[12, conn]
#             maxq = (maxq[1], maxq[2], maxq[3], gpuGlobalDataRest[12, conn])
#         end
#     end
#     return nothing
# end

# function min_q_values_kernel(gpuGlobalDataCommon, idx, minq)
#     minq = (
#                 gpuGlobalDataRest[9, idx],
#                 gpuGlobalDataRest[10, idx],
#                 gpuGlobalDataRest[11, idx],
#                 gpuGlobalDataRest[12, idx]
#             )
#     for iter in 9:28
#         conn = Int(gpuGlobalDataCommon[iter, idx])
#         if conn == 0.0
#             break
#         end
#         # currq = globaldata[itm].q
#         if minq[1] > gpuGlobalDataRest[9, conn]
#             minq = (gpuGlobalDataRest[9, conn], minq[2], minq[3], minq[4])
#         end
#         if minq[2] > gpuGlobalDataRest[10, conn]
#             minq = (minq[1], gpuGlobalDataRest[10, conn], minq[3], minq[4])
#         end
#         if minq[3] > gpuGlobalDataRest[11, conn]
#             minq = (minq[1], minq[2], gpuGlobalDataRest[11, conn], minq[4])
#         end
#         if minq[4] > gpuGlobalDataRest[12, conn]
#             minq = (minq[1], minq[2], minq[3], gpuGlobalDataRest[12, conn])
#         end
#     end
#     return nothing
# end

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

function qtilde_to_primitive_kernel(qtilde, gpuConfigData, gpuGlobalDataRest, idx)

    gamma = gpuConfigData[15]
    beta = -qtilde[4]*0.5
    temp = 0.5/beta
    u1 = qtilde[2]*temp
    u2 = qtilde[3]*temp

    temp2 = qtilde[1] + beta*(u1*u1 + u2*u2) - (CUDAnative.log(beta)/(gamma-1))
    rho = CUDAnative.exp(temp2)
    pr = rho*temp
    gpuGlobalDataRest[45, idx] = u1
    gpuGlobalDataRest[46, idx] = u2
    gpuGlobalDataRest[47, idx] = rho
    gpuGlobalDataRest[48, idx] = pr
    return nothing
end
