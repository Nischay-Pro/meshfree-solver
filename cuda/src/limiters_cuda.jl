
function venkat_limiter_kernel_qtilde(gpuGlobalDataFauxFixed, gpuGlobalDataRest, idx, gpuConfigData, numPoints, delx, dely, shared, qtilde_shared)
    thread_idx = threadIdx().x
    block_dim = blockDim().x
    epsigh = gpuConfigData[8] * gpuGlobalDataFauxFixed[idx + 5 * numPoints]
    epsi = epsigh*epsigh*epsigh
    gamma = gpuConfigData[15]
    # shared[thread_idx], shared[thread_idx + block_dim * 1], shared[thread_idx + block_dim * 2], shared[thread_idx + block_dim * 3] = 1,1,1,1
    del_pos = 0.0
    for i in 0:3
        qtilde_shared[thread_idx + block_dim * i] = 1.0
        q = gpuGlobalDataRest[idx, 9+i]
        del_neg = gpuGlobalDataRest[idx, 9+i] - 0.5*(delx * gpuGlobalDataRest[idx, 13+i] + dely * gpuGlobalDataRest[idx, 17+i]) - q
        if abs(del_neg) > 1e-5
            if del_neg > 0.0
                del_pos = gpuGlobalDataRest[idx, 21+i] - q
            elseif del_neg < 0.0
                del_pos = gpuGlobalDataRest[idx, 25+i] - q
            end
            num = (del_pos*del_pos) + (epsi*epsi)
            num = (num*del_neg) + 2 * (del_neg*del_neg*del_pos)

            den = (del_pos*del_pos) + (2 *del_neg*del_neg)
            den = den + (del_neg*del_pos) + (epsi*epsi)
            den *= del_neg

            temp = num/den
            if temp >= 1.0
                temp = 1.0
            end
            qtilde_shared[thread_idx + block_dim * i] = temp
        end
    end

    for i in 0:3
        temp = gpuGlobalDataRest[idx, 9 + i] - 0.5 * qtilde_shared[thread_idx + block_dim * i]*(delx * gpuGlobalDataRest[idx, 13+i] + dely * gpuGlobalDataRest[idx, 17+i])
        qtilde_shared[thread_idx + block_dim * i] = temp
    end
    beta = -qtilde_shared[thread_idx + block_dim * 3]*0.5
    temp = 0.5/beta
    u1 = qtilde_shared[thread_idx + block_dim * 1]*temp
    u2 = qtilde_shared[thread_idx + block_dim * 2]*temp

    temp2 = qtilde_shared[thread_idx] + beta*(u1*u1 + u2*u2) - (CUDA.log(beta)/(gamma-1))
    # rho = CUDA.exp(temp2)
    shared[thread_idx + block_dim * 4] = u1
    shared[thread_idx + block_dim * 5] = u2
    shared[thread_idx + block_dim * 6] = CUDA.exp(temp2)
    shared[thread_idx + block_dim * 7] = shared[thread_idx + block_dim * 6]*temp
    return nothing
end
