import math
from numba import cuda
import numba

@cuda.jit(device=True)
def venkat_limiter(qtilde_shared, globaldata, idx, VL_CONST, shared, delx, dely, gamma):
    epsi = VL_CONST * globaldata[idx]['min_dist']
    epsi = math.pow(epsi,3)
    for i in range(4):
        qtilde_shared[cuda.threadIdx.x + cuda.blockDim.x * i] = 1
        q = globaldata[idx]['q'][i]
        del_neg = globaldata[idx]['q'][i] - (0.5 * (delx * globaldata[idx]['dq'][0][i] + dely * globaldata[idx]['dq'][1][i])) - q
        
        if math.fabs(del_neg) <= 1e-5:
            qtilde_shared[cuda.threadIdx.x + cuda.blockDim.x * i] = 1

        elif math.fabs(del_neg) > 1e-5:
            if del_neg > 0:
                del_pos = globaldata[idx]['maxminq'][0][i] - q
            elif del_neg < 0:
                del_pos = globaldata[idx]['maxminq'][1][i] - q

            num = (del_pos*del_pos) + (epsi*epsi)
            num = num*del_neg + 2.0*del_neg*del_neg*del_pos

            den = del_pos*del_pos + 2.0*del_neg*del_neg
            den = den + del_neg*del_pos + epsi*epsi
            den = den*del_neg

            temp = num/den

            if temp < 1:
                qtilde_shared[cuda.threadIdx.x + cuda.blockDim.x * i] = temp

            else:
                qtilde_shared[cuda.threadIdx.x + cuda.blockDim.x * i] = 1 

    for i in range(4):
        qtilde_shared[cuda.threadIdx.x + cuda.blockDim.x * i] = globaldata[idx]['q'][i] - 0.5 * (qtilde_shared[cuda.threadIdx.x + cuda.blockDim.x * i] * (delx * globaldata[idx]['dq'][0][i] + dely * globaldata[idx]['dq'][1][i]))

    beta = - qtilde_shared[cuda.threadIdx.x + cuda.blockDim.x * 3] * 0.5

    temp = 0.5/beta

    u1 = qtilde_shared[cuda.threadIdx.x + cuda.blockDim.x * 1] * temp
    u2 = qtilde_shared[cuda.threadIdx.x + cuda.blockDim.x * 2] * temp

    temp1 = qtilde_shared[cuda.threadIdx.x] + beta * ( u1 * u1 + u2 * u2 )
    temp2 = temp1 - (math.log(beta)/(gamma-1))
    rho = math.exp(temp2)
    pr = rho * temp

    shared[cuda.threadIdx.x + cuda.blockDim.x * 4] = u1
    shared[cuda.threadIdx.x + cuda.blockDim.x * 5] = u2
    shared[cuda.threadIdx.x + cuda.blockDim.x * 6] = rho
    shared[cuda.threadIdx.x + cuda.blockDim.x * 7] = pr

@cuda.jit(device=True, inline=True)
def maximum(globaldata, idx, i, maxval, itm):
    maxval[i] = globaldata[idx]['q'][i]
    if maxval[i] < globaldata[itm]['q'][i]:
        maxval[i] = globaldata[itm]['q'][i]

@cuda.jit(device=True, inline=True)
def minimum(globaldata, idx, i, minval, itm):
    minval[i] = globaldata[idx]['q'][i]
    if minval[i] > globaldata[itm]['q'][i]:
        minval[i] = globaldata[itm]['q'][i]

@cuda.jit(device=True, inline=True)
def smallest_dist(globaldata, idx, min_dist):
    min_dist[0] = 10000

    for itm in globaldata[idx]['conn'][:globaldata[idx]['nbhs']]:
        dx = globaldata[idx]['x'] - globaldata[itm]['x']
        dy = globaldata[idx]['y'] - globaldata[itm]['y']
        ds = math.sqrt(dx * dx + dy * dy)
        
        if ds < min_dist[0]:
            min_dist[0] = ds

# def max_q_values(globaldata, idx, temp):

#     maxq = globaldata[idx]['q']

#     for itm in globaldata[idx]['conn']:
#         currq = globaldata[itm]['q']
#         for i in range(4):
#             if maxq[i] < currq[i]:
#                 maxq[i] = currq[i]
    
#     return maxq

# def min_q_values(globaldata, idx, temp):
#     minq = globaldata[idx]['q']

#     for itm in globaldata[idx]['conn']:
#         currq = globaldata[itm]['q']
#         for i in range(4):
#             if minq[i] > currq[i]:
#                 minq[i] = currq[i]
    
#     return minq