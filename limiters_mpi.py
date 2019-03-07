import math
import numpy as np

def venkat_limiter(qtilde, globaldata_local, globaldata_ghost, idx, configData):
    ghost = False
    VL_CONST = configData["core"]["vl_const"]
    phi = np.empty(4, dtype=np.float64)
    del_pos, del_neg = 0,0
    if idx in globaldata_local:
        itm = globaldata_local[idx]
    else:
        itm = globaldata_ghost[idx]
    for i in range(4):
        q = itm.q[i]
        del_neg = qtilde[i] - q
        if abs(del_neg) <= 1e-5:
            phi[i] = 1
        elif abs(del_neg) > 1e-5:
            if del_neg > 0:
                max_q = itm.maxq[i]
                del_pos = max_q - q
            elif del_neg < 0:
                min_q = itm.minq[i]
                del_pos = min_q - q
            ds = itm.ds
            epsi = VL_CONST * ds
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
    return phi



def maximum(globaldata_local, globaldata_ghost, idx, i, ghost):
    if not ghost:
        maxval = globaldata_local[idx].q[i]
        for itm in globaldata_local[idx].conn:
            try:
                if maxval < globaldata_local[itm].q[i]:
                    maxval = globaldata_local[itm].q[i]
            except:
                if maxval < globaldata_ghost[itm].q[i]:
                    maxval = globaldata_ghost[itm].q[i]
        return maxval
    else:
        maxval = globaldata_ghost[idx].q[i]
        for itm in globaldata_ghost[idx].conn:
            try:
                if maxval < globaldata_ghost[itm].q[i]:
                    maxval = globaldata_ghost[itm].q[i]
            except:
                if maxval < globaldata_local[itm].q[i]:
                    maxval = globaldata_local[itm].q[i]
        return maxval


def minimum(globaldata_local, globaldata_ghost, idx, i, ghost):
    if not ghost:
        minval = globaldata_local[idx].q[i]
        for itm in globaldata_local[idx].conn:
            try:
                if minval > globaldata_local[itm].q[i]:
                    minval = globaldata_local[itm].q[i]
            except:
                if minval > globaldata_ghost[itm].q[i]:
                    minval = globaldata_ghost[itm].q[i]
        return minval
    else:
        minval = globaldata_ghost[idx].q[i]
        for itm in globaldata_ghost[idx].conn:
            try:
                if minval > globaldata_ghost[itm].q[i]:
                    minval = globaldata_ghost[itm].q[i]
            except:
                if minval > globaldata_local[itm].q[i]:
                    minval = globaldata_local[itm].q[i]
        return minval

def smallest_dist(globaldata_local, globaldata_ghost, idx, ghost):
    min_dist = 10000
    if not ghost:
        for itm in globaldata_local[idx].conn:
            try:
                dx = globaldata_local[idx].x - globaldata_local[itm].x
                dy = globaldata_local[idx].y - globaldata_local[itm].y
            except:
                dx = globaldata_local[idx].x - globaldata_ghost[itm].x
                dy = globaldata_local[idx].y - globaldata_ghost[itm].y
            ds = math.sqrt(dx * dx + dy * dy)
            
            if ds < min_dist:
                min_dist = ds
    else:
        for itm in globaldata_ghost[idx].conn:
            try:
                dx = globaldata_ghost[idx].x - globaldata_ghost[itm].x
                dy = globaldata_ghost[idx].y - globaldata_ghost[itm].y
            except:
                dx = globaldata_ghost[idx].x - globaldata_local[itm].x
                dy = globaldata_ghost[idx].y - globaldata_local[itm].y
            ds = math.sqrt(dx * dx + dy * dy)
            
            if ds < min_dist:
                min_dist = ds
    return min_dist

def max_q_values(globaldata, idx):
    maxq = globaldata[idx].q

    for itm in globaldata[idx].conn:
        currq = globaldata[itm].q
        for i in range(4):
            if maxq[i] < currq[i]:
                maxq[i] = currq[i]
    
    return maxq

def min_q_values(globaldata, idx):
    minq = globaldata[idx].q

    for itm in globaldata[idx].conn:
        currq = globaldata[itm].q
        for i in range(4):
            if minq[i] > currq[i]:
                minq[i] = currq[i]
    
    return minq