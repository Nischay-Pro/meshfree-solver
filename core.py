import config
import math
import operator
import flux_residual
import state_update

def getInitialPrimitive(configData):
    rho_inf = float(configData["core"]["rho_inf"])
    mach = float(configData["core"]["mach"])
    machcos = mach * math.cos(calculateTheta(configData))
    machsin = mach * math.sin(calculateTheta(configData))
    pr_inf = float(configData["core"]["pr_inf"])
    primal = [rho_inf, machcos, machsin, pr_inf]
    return primal

    
def calculateTheta(configData):
    theta = math.radians(float(configData["core"]["aoa"]))
    return theta

def calculateNormals(left, right, mx, my):
    lx = left[0]
    ly = left[1]

    rx = right[0]
    ry = right[1]

    nx1 = my - ly
    nx2 = ry - my

    ny1 = mx - lx
    ny2 = rx - mx

    nx = 0.5*(nx1 + nx2)
    ny = 0.5*(ny1 + ny2)

    det = math.sqrt(nx*nx + ny*ny)

    nx = -nx/det
    ny = ny/det

    return (nx,ny)

def calculateConnectivity(globaldata, idx):
    ptInterest = globaldata[idx]
    currx = ptInterest.x
    curry = ptInterest.y
    nx = ptInterest.nx
    ny = ptInterest.ny

    flag = ptInterest.flag_1

    xpos_conn,xneg_conn,ypos_conn,yneg_conn = [],[],[],[]

    tx = ny
    ty = -nx

    for itm in ptInterest.conn:
        itmx = globaldata[itm].x
        itmy = globaldata[itm].y

        delx = itmx - currx
        dely = itmy - curry

        dels = delx*tx + dely*ty
        deln = delx*nx + dely*ny

        if dels <= 0:
            xpos_conn.append(itm)
        
        if dels >= 0:
            xneg_conn.append(itm)

        if flag == 1:
            if deln <= 0:
                ypos_conn.append(itm)
            
            if deln >= 0:
                yneg_conn.append(itm)

        elif flag == 0:
            yneg_conn.append(itm)
        
        elif flag == 2:
            ypos_conn.append(itm)
        
    return (xpos_conn, xneg_conn, ypos_conn, yneg_conn)

def fpi_solver(iter, globaldata, configData, wallindices, outerindices, interiorindices):
    globaldata = q_var_derivatives(globaldata, configData)
    globaldata = flux_residual.cal_flux_residual(globaldata, wallindices, outerindices, interiorindices, configData)
    globaldata = state_update.func_delta(globaldata, configData)
    globaldata = state_update.state_update(globaldata, wallindices, outerindices, interiorindices, configData, iter)

def q_var_derivatives(globaldata, configData):
    power = int(configData["core"]["power"])
    for idx,itm in enumerate(globaldata):
        if idx > 0:
            rho = itm.prim[0]
            u1 = itm.prim[1]
            u2 = itm.prim[2]
            pr = itm.prim[3]

            beta = 0.5 * (rho / pr)

            tempq = []

            tempq.append(math.log(rho) + (math.log(beta) * 2.5) - (beta * ((u1*u1) + (u2 * u2))))
            two_times_beta = 2 * beta

            tempq.append(two_times_beta * u1)
            tempq.append(two_times_beta * u2)
            tempq.append(-two_times_beta)

            globaldata[idx].q = tempq

    for idx,itm in enumerate(globaldata):
        if idx > 0:
            
            x_i = itm.x
            y_i = itm.y

            sum_delx_sqr = 0
            sum_dely_sqr = 0
            sum_delx_dely = 0

            sum_delx_delq = [0,0,0,0]
            sum_dely_delq = [0,0,0,0]

            for conn in itm.conn:
                

                x_k = globaldata[conn].x
                y_k = globaldata[conn].y

                delx = x_k - x_i
                dely = y_k - y_i

                dist = math.sqrt(delx*delx + dely*dely)
                weights = dist ** power


                sum_delx_sqr = sum_delx_sqr + (delx * delx * weights)
                sum_dely_sqr = sum_dely_sqr + (dely * dely * weights)

                sum_delx_dely = sum_delx_dely + (delx * dely * weights)

                sum_delx_delq[0] = sum_delx_delq[0] + (weights * delx * (globaldata[conn].q[0] - globaldata[idx].q[0]))
                sum_delx_delq[1] = sum_delx_delq[1] + (weights * delx * (globaldata[conn].q[1] - globaldata[idx].q[1]))
                sum_delx_delq[2] = sum_delx_delq[2] + (weights * delx * (globaldata[conn].q[2] - globaldata[idx].q[2]))
                sum_delx_delq[3] = sum_delx_delq[3] + (weights * delx * (globaldata[conn].q[3] - globaldata[idx].q[3]))

                sum_dely_delq[0] = sum_dely_delq[0] + (weights * dely * (globaldata[conn].q[0] - globaldata[idx].q[0]))
                sum_dely_delq[1] = sum_dely_delq[1] + (weights * dely * (globaldata[conn].q[1] - globaldata[idx].q[1]))
                sum_dely_delq[2] = sum_dely_delq[2] + (weights * dely * (globaldata[conn].q[2] - globaldata[idx].q[2]))
                sum_dely_delq[3] = sum_dely_delq[3] + (weights * dely * (globaldata[conn].q[3] - globaldata[idx].q[3]))

            det = (sum_delx_sqr * sum_dely_sqr) - (sum_delx_dely * sum_delx_dely)
            one_by_det = 1 / det

            tempdq = []

            sum_delx_delq = [i * sum_dely_sqr for i in sum_delx_delq]
            sum_dely_delq = [i * sum_delx_dely for i in sum_dely_delq]

            tempsumx = list(map(operator.sub, sum_delx_delq, sum_dely_delq))
            tempsumx = [i * one_by_det for i in tempsumx]

            sum_dely_delq = [i * sum_delx_sqr for i in sum_dely_delq]
            sum_delx_delq = [i * sum_delx_dely for i in sum_delx_delq]

            tempsumy = list(map(operator.sub, sum_dely_delq, sum_delx_delq))
            tempsumy = [i * one_by_det for i in tempsumy]

            tempdq.append(tempsumx)
            tempdq.append(tempsumy)

            globaldata[idx].dq = tempdq
    
    return globaldata

def qtilde_to_primitive(qtilde, configData):

    gamma = configData["core"]["gamma"]

    q1 = qtilde[0]
    q2 = qtilde[1]
    q3 = qtilde[2]
    q4 = qtilde[3]

    beta = -q4*0.5

    temp = 0.5/beta

    u1 = q2*temp
    u2 = q3*temp

    temp1 = q1 + beta*(u1*u1 + u2*u2)
    temp2 = temp1 - (math.log(beta)/(gamma-1))

    rho = math.exp(temp2)
    pr = rho*temp

    return (u1,u2,rho,pr)