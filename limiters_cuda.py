import math
from numba import cuda
import numba

@cuda.jit(device=True)
def venkat_limiter(qtilde, globaldata, idx, VL_CONST, phi, max_q, min_q, ds):
    del_pos = 0
    del_neg = 0
    for i in range(4):
        q = globaldata[idx]['q'][i]
        del_neg = qtilde[i] - q
        if math.fabs(del_neg) <= 1e-5:
            phi[i] = 1
        elif math.fabs(del_neg) > 1e-5:
            if del_neg > 0:
                max_q[0] = 0
                maximum(globaldata, idx, i, max_q)
                del_pos = max_q[0] - q
            elif del_neg < 0:
                min_q[0] = 0
                minimum(globaldata, idx, i, min_q)
                del_pos = min_q[0] - q

            ds[0] = 0
            smallest_dist(globaldata, idx, ds)
            epsi = VL_CONST * ds[0]
            epsi = math.pow(epsi,3)

            num = (del_pos*del_pos) + (epsi*epsi)
            num = num*del_neg + 2.0*del_neg*del_neg*del_pos

            den = del_pos*del_pos + 2.0*del_neg*del_neg
            den = den + del_neg*del_pos + epsi*epsi
            den = den*del_neg

            temp = num/den

            if temp < 1:
                phi[i] = temp
            else:
                phi[i] = 1

@cuda.jit(device=True)
def maximum(globaldata, idx, i, maxval):
    maxval[0] = globaldata[idx]['q'][i]
    for itm in globaldata[idx]['conn'][:globaldata[idx]['nbhs']]:
        if maxval[0] < globaldata[itm]['q'][i]:
            maxval[0] = globaldata[itm]['q'][i]

@cuda.jit(device=True)
def minimum(globaldata, idx, i, minval):
    minval[0] = globaldata[idx]['q'][i]
    for itm in globaldata[idx]['conn'][:globaldata[idx]['nbhs']]:
        if minval[0] > globaldata[itm]['q'][i]:
            minval[0] = globaldata[itm]['q'][i]

@cuda.jit(device=True)
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