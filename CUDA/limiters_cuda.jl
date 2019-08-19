function venkat_limiter_kernel(gpuGlobalDataFixedPoint, gpuGlobalDataRest, idx, gpuConfigData, delx, dely, shared, thread_idx, block_dim) 
    # @cuprintf("Type is %s", typeof(VL_CONST))
    epsigh = gpuConfigData[8] * gpuGlobalDataFixedPoint[idx].short_distance
    epsi = epsigh*epsigh*epsigh

    # shared[thread_idx], shared[thread_idx + block_dim * 1], shared[thread_idx + block_dim * 2], shared[thread_idx + block_dim * 3] = 1,1,1,1

    for i in 0:3
        shared[thread_idx + block_dim * i] = 1
        q = gpuGlobalDataRest[9+i, idx]
        del_neg = gpuGlobalDataRest[9+i, idx] - 0.5*(delx * gpuGlobalDataRest[13+i, idx] + dely * gpuGlobalDataRest[17+i, idx]) - q
        if abs(del_neg) > 1e-5
            del_pos = gpuGlobalDataRest[21+i, idx] - q
            if del_neg < 0
                del_pos = gpuGlobalDataRest[25+i, idx] - q
            end
            num = (del_pos*del_pos) + (epsi*epsi)
            num = (num*del_neg) + 2 * (del_neg*del_neg*del_pos)

            den = (del_pos*del_pos) + (2 *del_neg*del_neg)
            den += (del_neg*del_pos) + (epsi*epsi)
            den *= del_neg

            temp = num/den
            if temp > 1
                temp = 1
            end
            shared[thread_idx + block_dim * i] = temp
        end
    end
    return nothing
end


@inline function qtilde_to_primitive_kernel(qtilde, gpuConfigData, shared, thread_idx, block_dim)
    # # gamma = gpuConfigData[15]
    beta = -qtilde[4]*0.5
    temp = 0.5/beta
    u1 = qtilde[2]*temp
    u2 = qtilde[3]*temp

    temp2 = qtilde[1] + beta*(u1*u1 + u2*u2) - (CUDAnative.log(beta)/(gpuConfigData[15]-1))
    # rho = CUDAnative.exp(temp2)
    shared[thread_idx + block_dim * 4] = u1
    shared[thread_idx + block_dim * 5] = u2
    shared[thread_idx + block_dim * 6] = CUDAnative.exp(temp2)
    shared[thread_idx + block_dim * 7] = shared[thread_idx + block_dim * 6]*temp
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

# @inline function smallest_dist(globaldata, idx::Int64)
#     min_dist = 1000.0
#     for itm in globaldata[idx].conn
#         ds = hypot(globaldata[idx].x - globaldata[itm].x, globaldata[idx].y - globaldata[itm].y)
#         if ds < min_dist
#             min_dist = ds
#         end
#     end
#     globaldata[idx].short_distance = min_dist
# end
