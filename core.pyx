import config
import math
import operator
cimport flux_residual
import state_update
import objective_function
import numpy as np
cimport numpy as np

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
    currx = ptInterest.getx()
    curry = ptInterest.gety()
    nx = ptInterest.getnx()
    ny = ptInterest.getny()

    flag = ptInterest.get_flag_1()

    xpos_conn,xneg_conn,ypos_conn,yneg_conn = np.array([], dtype=np.long), np.array([], dtype=np.long), np.array([], dtype=np.long), np.array([], dtype=np.long)

    tx = ny
    ty = -nx

    for itm in ptInterest.get_conn():
        itmx = globaldata[itm].getx()
        itmy = globaldata[itm].gety()

        delx = itmx - currx
        dely = itmy - curry

        dels = delx*tx + dely*ty
        deln = delx*nx + dely*ny

        if dels <= 0:
            xpos_conn = np.append(xpos_conn, [itm])
        
        if dels >= 0:
            xneg_conn = np.append(xneg_conn, [itm])

        if flag == 1:
            if deln <= 0:
                ypos_conn = np.append(ypos_conn, [itm])
            
            if deln >= 0:
                yneg_conn = np.append(yneg_conn, [itm])

        elif flag == 0:
            yneg_conn = np.append(yneg_conn, [itm])
        
        elif flag == 2:
            ypos_conn = np.append(ypos_conn, [itm])
        
    return (xpos_conn, xneg_conn, ypos_conn, yneg_conn)

def fpi_solver(iter, globaldata, configData, wallindices, outerindices, interiorindices, res_old):
    globaldata = q_var_derivatives(globaldata, configData)
    globaldata = flux_residual.cal_flux_residual(globaldata, wallindices, outerindices, interiorindices, configData)
    globaldata = state_update.func_delta(globaldata, configData)
    globaldata, res_old = state_update.state_update(globaldata, wallindices, outerindices, interiorindices, configData, iter, res_old)
    # objective_function.compute_cl_cd_cm(globaldata, configData, wallindices)
    return res_old

def q_var_derivatives(globaldata, configData):
    power = int(configData["core"]["power"])
    for idx,itm in enumerate(globaldata):
        if idx > 0:
            tempitm = itm.getprim()
            rho = tempitm[0]
            u1 = tempitm[1]
            u2 = tempitm[2]
            pr = tempitm[3]

            beta = 0.5 * (rho / pr)

            tempq = np.array([])

            tempq = np.append(tempq, [math.log(rho) + (math.log(beta) * 2.5) - (beta * ((u1*u1) + (u2 * u2)))])
            two_times_beta = 2 * beta

            tempq = np.append(tempq, [two_times_beta * u1])
            tempq = np.append(tempq, [two_times_beta * u2])
            tempq = np.append(tempq, [-two_times_beta])

            globaldata[idx].setq(tempq)

    for idx,itm in enumerate(globaldata):
        if idx > 0:
            
            x_i = itm.getx()
            y_i = itm.gety()

            sum_delx_sqr = 0
            sum_dely_sqr = 0
            sum_delx_dely = 0

            sum_delx_delq = [0,0,0,0]
            sum_dely_delq = [0,0,0,0]

            for conn in itm.get_conn():
                

                x_k = globaldata[conn].getx()
                y_k = globaldata[conn].gety()

                delx = x_k - x_i
                dely = y_k - y_i

                dist = math.sqrt(delx*delx + dely*dely)
                weights = dist ** power


                sum_delx_sqr = sum_delx_sqr + (delx * delx * weights)
                sum_dely_sqr = sum_dely_sqr + (dely * dely * weights)

                sum_delx_dely = sum_delx_dely + (delx * dely * weights)

                sum_delx_delq[0] = sum_delx_delq[0] + (weights * delx * (globaldata[conn].getq()[0] - globaldata[idx].getq()[0]))
                sum_delx_delq[1] = sum_delx_delq[1] + (weights * delx * (globaldata[conn].getq()[1] - globaldata[idx].getq()[1]))
                sum_delx_delq[2] = sum_delx_delq[2] + (weights * delx * (globaldata[conn].getq()[2] - globaldata[idx].getq()[2]))
                sum_delx_delq[3] = sum_delx_delq[3] + (weights * delx * (globaldata[conn].getq()[3] - globaldata[idx].getq()[3]))

                sum_dely_delq[0] = sum_dely_delq[0] + (weights * dely * (globaldata[conn].getq()[0] - globaldata[idx].getq()[0]))
                sum_dely_delq[1] = sum_dely_delq[1] + (weights * dely * (globaldata[conn].getq()[1] - globaldata[idx].getq()[1]))
                sum_dely_delq[2] = sum_dely_delq[2] + (weights * dely * (globaldata[conn].getq()[2] - globaldata[idx].getq()[2]))
                sum_dely_delq[3] = sum_dely_delq[3] + (weights * dely * (globaldata[conn].getq()[3] - globaldata[idx].getq()[3]))

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

            globaldata[idx].setdq(np.array(tempdq))


    return globaldata