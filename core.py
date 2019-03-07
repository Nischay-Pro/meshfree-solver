import config
import math
import operator
import flux_residual
import state_update
import state_update_mpi
import objective_function
import numpy as np
from timeit import default_timer as timer
import sys
import time
import mpicore
import flux_residual_mpi
import limiters_mpi

def getInitialPrimitive(configData):
    rho_inf = float(configData["core"]["rho_inf"])
    mach = float(configData["core"]["mach"])
    machcos = mach * math.cos(calculateTheta(configData))
    machsin = mach * math.sin(calculateTheta(configData))
    pr_inf = float(configData["core"]["pr_inf"])
    primal = np.array([rho_inf, machcos, machsin, pr_inf])
    return primal

def getInitialPrimitive2(configData):
    dataman = open("prim_soln_clean")
    data = dataman.read()
    data = data.split("\n")
    finaldata = []
    for idx,itm in enumerate(data):
        try:
            da = itm.split(" ")
            da = list(map(float, da))
            finaldata.append(da)
        except:
            print(idx)
    return finaldata
    
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

def calculateConnectivity(globaldata, idx, configData):
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

        if flag == configData["point"]["interior"]:
            if deln <= 0:
                ypos_conn.append(itm)
            
            if deln >= 0:
                yneg_conn.append(itm)

        elif flag == configData["point"]["wall"]:
            yneg_conn.append(itm)
        
        elif flag == configData["point"]["outer"]:
            ypos_conn.append(itm)
        
    return (xpos_conn, xneg_conn, ypos_conn, yneg_conn)

def calculateConnectivityMPI(globaldata_local, idx, configData, globaldata_ghost):
    ptInterest = globaldata_local[idx]
    currx = ptInterest.x
    curry = ptInterest.y
    nx = ptInterest.nx
    ny = ptInterest.ny

    flag = ptInterest.flag_1

    xpos_conn,xneg_conn,ypos_conn,yneg_conn = [],[],[],[]

    tx = ny
    ty = -nx

    for itm in ptInterest.conn:
        if itm in globaldata_local.keys():
            itmx = globaldata_local[itm].x
            itmy = globaldata_local[itm].y
        else:
            itmx = globaldata_ghost[itm].x
            itmy = globaldata_ghost[itm].y

        delx = itmx - currx
        dely = itmy - curry

        dels = delx*tx + dely*ty
        deln = delx*nx + dely*ny

        if dels <= 0:
            xpos_conn.append(itm)
        
        if dels >= 0:
            xneg_conn.append(itm)

        if flag == configData["point"]["interior"]:
            if deln <= 0:
                ypos_conn.append(itm)
            
            if deln >= 0:
                yneg_conn.append(itm)

        elif flag == configData["point"]["wall"]:
            yneg_conn.append(itm)
        
        elif flag == configData["point"]["outer"]:
            ypos_conn.append(itm)
        
    return (xpos_conn, xneg_conn, ypos_conn, yneg_conn)


def fpi_solver(iter, globaldata, configData, wallindices, outerindices, interiorindices, res_old):
    np.set_printoptions(precision=13)
    for i in range(1, iter):
        globaldata = q_var_derivatives(globaldata, configData)
        globaldata = flux_residual.cal_flux_residual(globaldata, wallindices, outerindices, interiorindices, configData)
        globaldata = state_update.func_delta(globaldata, configData)
        globaldata, res_old = state_update.state_update(globaldata, wallindices, outerindices, interiorindices, configData, i, res_old)
        objective_function.compute_cl_cd_cm(globaldata, configData, wallindices)
    return res_old, globaldata    

def fpi_solver_mpi(iter, globaldata_local, configData, globaldata_ghost, res_old, wallindices, outerindices, interiorindices, comm, globaldata_table):
    np.set_printoptions(precision=13)
    foreign_communicators = None
    ds = False
    for i in range(1, iter):
        globaldata_local, globaldata_ghost = q_var_derivatives_mpi(globaldata_local, globaldata_ghost, configData, ds=ds)
        comm.Barrier()
        if ds == False:
            globaldata_ghost, foreign_communicators = mpicore.sync_ghost(globaldata_local, globaldata_ghost, globaldata_table, comm, foreign_communicators, [0, 1, 2, 3])
            ds = True
        else:
            globaldata_ghost, foreign_communicators = mpicore.sync_ghost(globaldata_local, globaldata_ghost, globaldata_table, comm, foreign_communicators, [0, 1, 2])
        globaldata_local = flux_residual_mpi.cal_flux_residual_mpi(globaldata_local, globaldata_ghost, wallindices, outerindices, interiorindices, configData)
        globaldata_local = state_update_mpi.func_delta_mpi(globaldata_local, globaldata_ghost, configData)
        globaldata_local, res_old = state_update_mpi.state_update_mpi(globaldata_local, wallindices, outerindices, interiorindices, configData, i, res_old, comm)
        comm.Barrier()
        globaldata_ghost, foreign_communicators = mpicore.sync_ghost(globaldata_local, globaldata_ghost, globaldata_table, comm, foreign_communicators, [4])
        # objective_function.compute_cl_cd_cm(globaldata_local, configData, wallindices, comm=comm)
        # if comm.rank == 0:
        #     np.set_printoptions(precision=17)
        #     print(globaldata_local[1].x)
        #     print(globaldata_local[1].y)
        #     print(globaldata_local[1].dq)
        #     print(globaldata_local[1].prim)
    return res_old, globaldata_local   

def q_var_derivatives(globaldata, configData):
    power = int(configData["core"]["power"])
    for idx,itm in enumerate(globaldata):
        if idx > 0:
            rho = itm.prim[0]
            u1 = itm.prim[1]
            u2 = itm.prim[2]
            pr = itm.prim[3]

            beta = 0.5 * (rho / pr)

            tempq = np.zeros(4, dtype=np.float64)

            tempq[0] = (math.log(rho) + (math.log(beta) * 2.5) - (beta * ((u1*u1) + (u2 * u2))))
            two_times_beta = 2 * beta

            tempq[1] = (two_times_beta * u1)
            tempq[2] = (two_times_beta * u2)
            tempq[3] = -two_times_beta

            globaldata[idx].q = tempq

    for idx,itm in enumerate(globaldata):
        if idx > 0:
            
            x_i = itm.x
            y_i = itm.y

            sum_delx_sqr = 0
            sum_dely_sqr = 0
            sum_delx_dely = 0

            sum_delx_delq = np.zeros(4, dtype=np.float64)
            sum_dely_delq = np.zeros(4, dtype=np.float64)

            for conn in itm.conn:
                

                x_k = globaldata[conn].x
                y_k = globaldata[conn].y

                delx = x_k - x_i
                dely = y_k - y_i

                dist = math.sqrt(delx*delx + dely*dely)
                weights = dist ** power


                sum_delx_sqr = sum_delx_sqr + ((delx * delx) * weights)
                sum_dely_sqr = sum_dely_sqr + ((dely * dely) * weights)

                sum_delx_dely = sum_delx_dely + ((delx * dely) * weights)

                sum_delx_delq = sum_delx_delq + (weights * delx * (globaldata[conn].q - globaldata[idx].q))

                sum_dely_delq = sum_dely_delq + (weights * dely * (globaldata[conn].q - globaldata[idx].q))

            det = (sum_delx_sqr * sum_dely_sqr) - (sum_delx_dely * sum_delx_dely)
            one_by_det = 1 / det

            tempdq = np.zeros(2, dtype=np.float64)

            sum_delx_delq1 = sum_delx_delq * sum_dely_sqr
            sum_dely_delq1 = sum_dely_delq * sum_delx_dely

            tempsumx = one_by_det * (sum_delx_delq1 - sum_dely_delq1)

            sum_dely_delq2 = sum_dely_delq * sum_delx_sqr

            sum_delx_delq2 = sum_delx_delq * sum_delx_dely

            tempsumy = one_by_det * (sum_dely_delq2 - sum_delx_delq2)

            tempdq = np.array([tempsumx, tempsumy], dtype=np.float64)

            globaldata[idx].dq = tempdq


    return globaldata

def q_var_derivatives_mpi(globaldata_local, globaldata_ghost, configData, ds=False):
    power = int(configData["core"]["power"])
    for idx in globaldata_local.keys():
        if idx > 0:
            itm = globaldata_local[idx]
            rho = itm.prim[0]
            u1 = itm.prim[1]
            u2 = itm.prim[2]
            pr = itm.prim[3]

            beta = 0.5 * (rho / pr)

            tempq = np.zeros(4, dtype=np.float64)

            tempq[0] = (math.log(rho) + (math.log(beta) * 2.5) - (beta * ((u1*u1) + (u2 * u2))))
            two_times_beta = 2 * beta

            tempq[1] = (two_times_beta * u1)
            tempq[2] = (two_times_beta * u2)
            tempq[3] = -two_times_beta

            globaldata_local[idx].q = tempq

            if ds == False:
                globaldata_local[idx].ds = limiters_mpi.smallest_dist(globaldata_local, globaldata_ghost, idx, False)

    for idx in globaldata_ghost.keys():
        if idx > 0:
            itm = globaldata_ghost[idx]
            rho = itm.prim[0]
            u1 = itm.prim[1]
            u2 = itm.prim[2]
            pr = itm.prim[3]

            beta = 0.5 * (rho / pr)

            tempq = np.zeros(4, dtype=np.float64)

            tempq[0] = (math.log(rho) + (math.log(beta) * 2.5) - (beta * ((u1*u1) + (u2 * u2))))
            two_times_beta = 2 * beta

            tempq[1] = (two_times_beta * u1)
            tempq[2] = (two_times_beta * u2)
            tempq[3] = -two_times_beta

            globaldata_ghost[idx].q = tempq

    for idx in globaldata_local.keys():
        if idx > 0:
            globaldata_local[idx].maxq = np.zeros(4, dtype=np.float64) 
            globaldata_local[idx].minq = np.zeros(4, dtype=np.float64)
            for i in range(4): 
                globaldata_local[idx].maxq[i] = limiters_mpi.maximum(globaldata_local, globaldata_ghost, idx, i, False)
                globaldata_local[idx].minq[i] = limiters_mpi.minimum(globaldata_local, globaldata_ghost, idx, i, False)

            itm = globaldata_local[idx]
            x_i = itm.x
            y_i = itm.y

            sum_delx_sqr = 0
            sum_dely_sqr = 0
            sum_delx_dely = 0

            sum_delx_delq = np.zeros(4, dtype=np.float64)
            sum_dely_delq = np.zeros(4, dtype=np.float64)

            for conn in itm.conn:
                ghost = False
                if conn in globaldata_local.keys():
                    x_k = globaldata_local[conn].x
                    y_k = globaldata_local[conn].y
                else:
                    x_k = globaldata_ghost[conn].x
                    y_k = globaldata_ghost[conn].y
                    ghost = True

                delx = x_k - x_i
                dely = y_k - y_i

                dist = math.sqrt(delx*delx + dely*dely)
                weights = dist ** power


                sum_delx_sqr = sum_delx_sqr + ((delx * delx) * weights)
                sum_dely_sqr = sum_dely_sqr + ((dely * dely) * weights)

                sum_delx_dely = sum_delx_dely + ((delx * dely) * weights)

                if not ghost:
                    sum_delx_delq = sum_delx_delq + (weights * delx * (globaldata_local[conn].q - globaldata_local[idx].q))

                    sum_dely_delq = sum_dely_delq + (weights * dely * (globaldata_local[conn].q - globaldata_local[idx].q))

                else:
                    sum_delx_delq = sum_delx_delq + (weights * delx * (globaldata_ghost[conn].q - globaldata_local[idx].q))

                    sum_dely_delq = sum_dely_delq + (weights * dely * (globaldata_ghost[conn].q - globaldata_local[idx].q))

            det = (sum_delx_sqr * sum_dely_sqr) - (sum_delx_dely * sum_delx_dely)
            one_by_det = 1 / det

            tempdq = np.zeros(2, dtype=np.float64)

            sum_delx_delq1 = sum_delx_delq * sum_dely_sqr
            sum_dely_delq1 = sum_dely_delq * sum_delx_dely

            tempsumx = one_by_det * (sum_delx_delq1 - sum_dely_delq1)

            sum_dely_delq2 = sum_dely_delq * sum_delx_sqr

            sum_delx_delq2 = sum_delx_delq * sum_delx_dely

            tempsumy = one_by_det * (sum_dely_delq2 - sum_delx_delq2)

            tempdq = np.array([tempsumx, tempsumy], dtype=np.float64)

            globaldata_local[idx].dq = tempdq


    return globaldata_local, globaldata_ghost

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
