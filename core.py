import config
import math
import operator
import flux_residual
import state_update
import objective_function
import numpy as np
from timeit import default_timer as timer
import numba
from numba import cuda
import convert
from numba import vectorize, float64
from cuda_func import add, subtract, multiply

def getInitialPrimitive(configData):
    rho_inf = float(configData["core"]["rho_inf"])
    mach = float(configData["core"]["mach"])
    machcos = mach * math.cos(calculateTheta(configData))
    machsin = mach * math.sin(calculateTheta(configData))
    pr_inf = float(configData["core"]["pr_inf"])
    primal = [rho_inf, machcos, machsin, pr_inf]
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

        if flag == 2:
            if deln <= 0:
                ypos_conn.append(itm)
            
            if deln >= 0:
                yneg_conn.append(itm)

        elif flag == 1:
            yneg_conn.append(itm)
        
        elif flag == 3:
            ypos_conn.append(itm)
        
    return (xpos_conn, xneg_conn, ypos_conn, yneg_conn)

def fpi_solver(iter, globaldata, configData, wallindices, outerindices, interiorindices, res_old):
    if not cuda.is_available():
        globaldata = q_var_derivatives(globaldata, configData)
        # with open('test_cpu', 'w+') as the_file:
        #     for idx in range(len(globaldata)):
        #         if idx > 0:
        #             itm = globaldata[idx]
        #             the_file.write(str(itm.q) + "\n")
        globaldata = flux_residual.cal_flux_residual(globaldata, wallindices, outerindices, interiorindices, configData)
        globaldata = state_update.func_delta(globaldata, configData)
        globaldata, res_old = state_update.state_update(globaldata, wallindices, outerindices, interiorindices, configData, iter, res_old)
        objective_function.compute_cl_cd_cm(globaldata, configData, wallindices)
        return res_old, globaldata
    else:
        fpi_solver_cuda(iter, globaldata, configData, wallindices, outerindices, interiorindices, res_old)
        return res_old, globaldata


def fpi_solver_cuda(iter, globaldata, configData, wallindices, outerindices, interiorindices, res_old):
    stream = cuda.stream()
    globaldata_gpu = convert.convert_globaldata_to_gpu_globaldata(globaldata)
    with stream.auto_synchronize():
        globaldata_gpu = cuda.to_device(globaldata_gpu, stream)
        threadsperblock = (128, 1)
        blockspergrid_x = math.ceil(len(globaldata) / threadsperblock[0])
        blockspergrid_y = math.ceil(1)
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        q_var_cuda_kernel[blockspergrid, threadsperblock](globaldata_gpu)
        cuda.synchronize()
        q_var_derivatives_cuda_kernel[blockspergrid, threadsperblock](globaldata_gpu, configData['core']['power'])
        cuda.synchronize()
        flux_residual.cal_flux_residual_cuda_kernel[blockspergrid, threadsperblock](globaldata_gpu, configData['core']['power'], configData['core']['vl_const'], configData['core']['gamma'])
        cuda.synchronize()
        temp = globaldata_gpu.copy_to_host()
    globaldata = convert.convert_gpu_globaldata_to_globaldata(temp)
    # with open('test_gpu', 'w+') as the_file:
    #     for idx in range(len(globaldata)):
    #         if idx > 0:
    #             itm = globaldata[idx]
    #             the_file.write(str(itm.q) + "\n")
    globaldata = state_update.func_delta(globaldata, configData)
    globaldata, res_old = state_update.state_update(globaldata, wallindices, outerindices, interiorindices, configData, iter, res_old)
    objective_function.compute_cl_cd_cm(globaldata, configData, wallindices)
    return res_old, globaldata
        
    

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

@cuda.jit()
def q_var_cuda_kernel(globaldata):
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x
    idx =  bx * bw + tx
    if idx > 0 and idx < len(globaldata):
        itm = globaldata[idx]
        rho = itm['prim'][0]
        u1 = itm['prim'][1]
        u2 = itm['prim'][2]
        pr = itm['prim'][3]

        beta = 0.5 * (rho / pr)

        tempq = cuda.local.array((4), dtype=numba.float64)

        tempq[0] = (math.log(rho) + (math.log(beta) * 2.5) - (beta * ((u1*u1) + (u2 * u2))))
        two_times_beta = 2 * beta

        tempq[1] = (two_times_beta * u1)
        tempq[2] = (two_times_beta * u2)
        tempq[3] = -two_times_beta

        globaldata[idx]['q'][0] = tempq[0]
        globaldata[idx]['q'][1] = tempq[1]
        globaldata[idx]['q'][2] = tempq[2]
        globaldata[idx]['q'][3] = tempq[3]
        

@cuda.jit()
def q_var_derivatives_cuda_kernel(globaldata, power):
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x
    idx =  bx * bw + tx
    if idx > 0 and idx < len(globaldata):

        itm = globaldata[idx]
        x_i = itm['x']
        y_i = itm['y']

        sum_delx_sqr = 0
        sum_dely_sqr = 0
        sum_delx_dely = 0

        sum_delx_delq = cuda.local.array((4), dtype=numba.float64)
        sum_dely_delq = cuda.local.array((4), dtype=numba.float64)

        sum_delx_delq[0] = 0
        sum_delx_delq[1] = 0
        sum_delx_delq[2] = 0
        sum_delx_delq[3] = 0

        sum_dely_delq[0] = 0
        sum_dely_delq[1] = 0
        sum_dely_delq[2] = 0
        sum_dely_delq[3] = 0

        for conn in itm['conn'][:itm['nbhs']]:


            x_k = globaldata[conn]['x']
            y_k = globaldata[conn]['y']

            delx = x_k - x_i
            dely = y_k - y_i

            dist = math.sqrt(delx*delx + dely*dely)
            weights = dist ** power


            sum_delx_sqr = sum_delx_sqr + ((delx * delx) * weights)
            sum_dely_sqr = sum_dely_sqr + ((dely * dely) * weights)

            sum_delx_dely = sum_delx_dely + ((delx * dely) * weights)

            temp = cuda.local.array((4), dtype=numba.float64)

            temp[0] = 0
            temp[1] = 0
            temp[2] = 0
            temp[3] = 0

            subtract(globaldata[conn]['q'], globaldata[idx]['q'], temp)
            multiply((weights * delx), temp, temp)
            add(sum_delx_delq, temp, sum_delx_delq)

            temp[0] = 0
            temp[1] = 0
            temp[2] = 0
            temp[3] = 0

            subtract(globaldata[conn]['q'], globaldata[idx]['q'], temp)
            multiply((weights * dely), temp, temp)
            add(sum_dely_delq, temp, sum_dely_delq)

        det = (sum_delx_sqr * sum_dely_sqr) - (sum_delx_dely * sum_delx_dely)

        one_by_det = 1 / det

        sum_delx_delq1 = cuda.local.array((4), dtype=numba.float64)
        sum_dely_delq1 = cuda.local.array((4), dtype=numba.float64)

        sum_delx_delq1[0] = 0
        sum_delx_delq1[1] = 0
        sum_delx_delq1[2] = 0
        sum_delx_delq1[3] = 0

        sum_dely_delq1[0] = 0
        sum_dely_delq1[1] = 0
        sum_dely_delq1[2] = 0
        sum_dely_delq1[3] = 0
        
        multiply(sum_dely_sqr, sum_delx_delq, sum_delx_delq1)
        multiply(sum_delx_dely, sum_dely_delq, sum_dely_delq1)

        subtract(sum_delx_delq1, sum_dely_delq1, sum_delx_delq1)
        multiply(one_by_det, sum_delx_delq1, sum_delx_delq1)

        tempsumx = cuda.local.array((4), dtype=numba.float64)


        tempsumx[0] = sum_delx_delq1[0]
        tempsumx[1] = sum_delx_delq1[1]
        tempsumx[2] = sum_delx_delq1[2]
        tempsumx[3] = sum_delx_delq1[3]

        sum_delx_delq1[0] = 0
        sum_delx_delq1[1] = 0
        sum_delx_delq1[2] = 0
        sum_delx_delq1[3] = 0

        sum_dely_delq1[0] = 0
        sum_dely_delq1[1] = 0
        sum_dely_delq1[2] = 0
        sum_dely_delq1[3] = 0

        multiply(sum_delx_sqr, sum_dely_delq, sum_delx_delq1)
        multiply(sum_delx_dely, sum_delx_delq, sum_dely_delq1)

        subtract(sum_delx_delq1, sum_dely_delq1, sum_dely_delq1)
        multiply(one_by_det, sum_dely_delq1, sum_dely_delq1)

        tempsumy = cuda.local.array((4), dtype=numba.float64)

        tempsumy[0] = sum_dely_delq1[0]
        tempsumy[1] = sum_dely_delq1[1]
        tempsumy[2] = sum_dely_delq1[2]
        tempsumy[3] = sum_dely_delq1[3]
    
        globaldata[idx]['dq'][0][0] = tempsumx[0]
        globaldata[idx]['dq'][0][1] = tempsumx[1]
        globaldata[idx]['dq'][0][2] = tempsumx[2]
        globaldata[idx]['dq'][0][3] = tempsumx[3]

        globaldata[idx]['dq'][1][0] = tempsumy[0]
        globaldata[idx]['dq'][1][1] = tempsumy[1]
        globaldata[idx]['dq'][1][2] = tempsumy[2]
        globaldata[idx]['dq'][1][3] = tempsumy[3]


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