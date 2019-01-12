import math

def venkat_limiter(qtilde, globaldata, idx, configData):
    VL_CONST = configData["core"]["vl_const"]
    phi = []
    del_pos, del_neg = 0,0
    for i in range(4):
        q = globaldata[idx].getq()[i]
        del_neg = qtilde[i] - q
        if abs(del_neg) <= 1e-5:
            phi.append(1)
        elif abs(del_neg) > 1e-5:
            if del_neg > 0:
                max_q = maximum(globaldata, idx, i)
                del_pos = max_q - q
            elif del_neg < 0:
                min_q = minimum(globaldata, idx, i)
                del_pos = min_q - q

            ds = smallest_dist(globaldata,idx)
            epsi = VL_CONST * ds
            epsi = pow(epsi,3)

            num = (del_pos*del_pos) + (epsi*epsi)
            num = num*del_neg + 2.0*del_neg*del_neg*del_pos

            den = del_pos*del_pos + 2.0*del_neg*del_neg
            den = den + del_neg*del_pos + epsi*epsi
            den = den*del_neg

            temp = num/den

            if temp < 1:
                phi.append(temp)
            else:
                phi.append(1)
    
    return phi



def maximum(globaldata, idx, i):
    maxval = globaldata[idx].getq()[i]
    for itm in globaldata[idx].conn:
        if maxval < globaldata[itm].getq()[i]:
            maxval = globaldata[itm].getq()[i]
    return maxval


def minimum(globaldata, idx, i):
    minval = globaldata[idx].getq()[i]
    for itm in globaldata[idx].conn:
        if minval > globaldata[itm].getq()[i]:
            minval = globaldata[itm].getq()[i]
    return minval

def smallest_dist(globaldata, idx):
    min_dist = 10000

    for itm in globaldata[idx].conn:
        dx = globaldata[idx].getx() - globaldata[itm].getx()
        dy = globaldata[idx].gety() - globaldata[itm].gety()
        ds = math.sqrt(dx * dx + dy * dy)
        
        if ds < min_dist:
            min_dist = ds

    return min_dist

def max_q_values(globaldata, idx):
    maxq = globaldata[idx].getq()

    for itm in globaldata[idx].conn:
        currq = globaldata[itm].getq()
        for i in range(4):
            if maxq[i] < currq[i]:
                maxq[i] = currq[i]
    
    return maxq

def min_q_values(globaldata, idx):
    minq = globaldata[idx].getq()

    for itm in globaldata[idx].conn:
        currq = globaldata[itm].getq()
        for i in range(4):
            if minq[i] > currq[i]:
                minq[i] = currq[i]
    
    return minq