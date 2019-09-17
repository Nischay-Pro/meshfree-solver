import config
import math
import operator
import flux_residual
import state_update
import state_update_cuda
import objective_function
import numpy as np
from timeit import default_timer as timer
import time
import numba
from numba import cuda
import convert
from numba import vectorize, float64
from cuda_func import add, subtract, multiply, sum_reduce, equalize
import os
import limiters_cuda
import helper

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

@cuda.jit(device=True)
def calculateThetaCuda(aoa, value):
    pi = 3.14159265358979323846
    value[0] = (pi/180) * aoa

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

def fpi_solver(iter, globaldata, configData, wallindices, outerindices, interiorindices, res_old):
    if not cuda.is_available():
        a = time.time()
        for i in range(1, iter):
            globaldata = q_var_derivatives(globaldata, configData)
            globaldata = flux_residual.cal_flux_residual(globaldata, wallindices, outerindices, interiorindices, configData)
            globaldata = state_update.func_delta(globaldata, configData)
            globaldata, res_old = state_update.state_update(globaldata, wallindices, outerindices, interiorindices, configData, i, res_old)
            objective_function.compute_cl_cd_cm(globaldata, configData, wallindices)
        b = time.time()
        with open('grid_{}.txt'.format(len(globaldata)), 'a+') as the_file:
            the_file.write("Runtime: {}\nIterations: {}\n".format((b - a), iter))
        return res_old, globaldata
    else:
        return fpi_solver_cuda(iter, globaldata, configData, wallindices, outerindices, interiorindices, res_old)


def fpi_solver_cuda(iter, globaldata, configData, wallindices, outerindices, interiorindices, res_old):
    stream = cuda.stream()
    print("Converting Globaldata to GPU")
    try:
        os.remove("residue")
    except:
        pass
    x_gpu, y_gpu, left_gpu, right_gpu, flag_1_gpu, flag_2_gpu, nbhs_gpu, conn_gpu, nx_gpu, ny_gpu, prim_gpu, prim_old_gpu, flux_res_gpu, q_gpu, dq_gpu, xpos_nbhs_gpu, xneg_nbhs_gpu, ypos_nbhs_gpu, yneg_nbhs_gpu, xpos_conn_gpu, xneg_conn_gpu, ypos_conn_gpu, yneg_conn_gpu, delta_gpu, min_dist_gpu, maxminq = convert.convert_globaldata_to_gpu_globaldata_new(globaldata)
    sum_res_sqr = np.zeros(len(globaldata), dtype=np.float64)
    a = time.time()
    if configData['core']['ssprk']:
        rks = 5
        eu = 1
    else:
        rks = 1
        eu = 2
    with stream.auto_synchronize():
        print("Pushing GPU Globaldata to GPU")
        x_gpu = cuda.to_device(x_gpu, stream)
        y_gpu = cuda.to_device(y_gpu, stream)
        left_gpu = cuda.to_device(left_gpu, stream)
        right_gpu = cuda.to_device(right_gpu, stream)
        flag_1_gpu = cuda.to_device(flag_1_gpu, stream)
        flag_2_gpu = cuda.to_device(flag_2_gpu, stream)
        nbhs_gpu = cuda.to_device(nbhs_gpu, stream)
        conn_gpu = cuda.to_device(conn_gpu, stream)
        nx_gpu = cuda.to_device(nx_gpu, stream)
        ny_gpu = cuda.to_device(ny_gpu, stream)
        prim_gpu = cuda.to_device(prim_gpu, stream)
        prim_old_gpu = cuda.to_device(prim_old_gpu, stream)
        flux_res_gpu = cuda.to_device(flux_res_gpu, stream)
        q_gpu = cuda.to_device(q_gpu, stream)
        dq_gpu = cuda.to_device(dq_gpu, stream)
        xpos_nbhs_gpu = cuda.to_device(xpos_nbhs_gpu, stream)
        xneg_nbhs_gpu = cuda.to_device(xneg_nbhs_gpu, stream)
        ypos_nbhs_gpu = cuda.to_device(ypos_nbhs_gpu, stream)
        yneg_nbhs_gpu = cuda.to_device(yneg_nbhs_gpu, stream)
        xpos_conn_gpu = cuda.to_device(xpos_conn_gpu, stream)
        xneg_conn_gpu = cuda.to_device(xneg_conn_gpu, stream)
        ypos_conn_gpu = cuda.to_device(ypos_conn_gpu, stream)
        yneg_conn_gpu = cuda.to_device(yneg_conn_gpu, stream)
        delta_gpu = cuda.to_device(delta_gpu, stream)
        min_dist_gpu = cuda.to_device(min_dist_gpu, stream)
        maxminq_gpu = cuda.to_device(maxminq, stream)

        sum_res_sqr_gpu = cuda.to_device(sum_res_sqr, stream)
        threadsperblock = (int(configData['core']['blockGridX']), 1)
        blockspergrid_x = math.ceil(len(globaldata) / threadsperblock[0])
        blockspergrid_y = math.ceil(1)
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        for i in range(1, iter):
            state_update_cuda.func_delta_cuda_kernel[blockspergrid, threadsperblock](prim_gpu, prim_old_gpu, x_gpu, y_gpu, conn_gpu, nbhs_gpu, delta_gpu, float(configData["core"]["cfl"]))
            if i == 1:
                print("Compiling CUDA Kernel. This might take a while...")
                c = time.time()
            for rk in range(1, rks):
                q_var_cuda_kernel[blockspergrid, threadsperblock](prim_gpu, q_gpu)
                q_var_derivatives_cuda_kernel[blockspergrid, threadsperblock](x_gpu, y_gpu, maxminq_gpu, conn_gpu, nbhs_gpu, q_gpu, dq_gpu, float(configData['core']['power']))
                flux_residual.cal_flux_residual_cuda_kernel[blockspergrid, threadsperblock](x_gpu, y_gpu, nx_gpu, ny_gpu, flag_1_gpu, min_dist_gpu, nbhs_gpu, conn_gpu, xpos_nbhs_gpu, xpos_conn_gpu, xneg_nbhs_gpu, xneg_conn_gpu, ypos_nbhs_gpu, ypos_conn_gpu, yneg_nbhs_gpu, yneg_conn_gpu, prim_gpu, q_gpu, maxminq_gpu, dq_gpu, flux_res_gpu, float(configData['core']['power']), int(configData['core']['vl_const']), float(configData['core']['gamma']), int(configData["point"]["wall"]), int(configData["point"]["interior"]), int(configData["point"]["outer"]))
                state_update_cuda.state_update_cuda[blockspergrid, threadsperblock](x_gpu, y_gpu, nx_gpu, ny_gpu, flag_1_gpu, nbhs_gpu, conn_gpu, prim_gpu, prim_old_gpu, delta_gpu, flux_res_gpu, float(configData["core"]["mach"]), float(configData["core"]["gamma"]), float(configData["core"]["pr_inf"]), float(configData["core"]["rho_inf"]), float(configData["core"]["aoa"]), sum_res_sqr_gpu, int(configData["point"]["wall"]), int(configData["point"]["interior"]), int(configData["point"]["outer"]), rk, eu)
            if i == 1:
                d = time.time()
            temp_gpu = sum_reduce(sum_res_sqr_gpu)
            residue = math.sqrt(temp_gpu) / (len(x_gpu) - 1)
            if i <= 2:
                res_old = residue
                residue = 0
            else:
                residue = math.log10(residue / res_old)
            print("Iteration: %s Residue: %s" % (str(i), residue))
            with open('residue', 'a+') as the_file:
                the_file.write("%s %s\n" % (i, residue))
        temp = x_gpu.copy_to_host()
        if configData["core"]["debug"]:
            sum_res_sqr = sum_res_sqr_gpu.copy_to_host()
    b = time.time()
    with open('grid_{}.txt'.format(len(globaldata)), 'a+') as the_file:
        the_file.write("Block Dimensions: ({}, 1)\nRuntime: {}\n".format(int(configData['core']['blockGridX']), (b - a - (d - c))))
    if configData["core"]["profile"]:
        exit()
    globaldata = convert.convert_gpu_globaldata_to_globaldata(temp)
    objective_function.compute_cl_cd_cm(globaldata, configData, wallindices)
    if configData["core"]["debug"]:
        helper.printPrimitive(globaldata)
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

@cuda.jit(inline=True)
def q_var_cuda_kernel(prim, q):
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x
    idx =  bx * bw + tx
    if idx > 0 and idx < len(prim):
        itm = prim[idx]
        rho = itm[0]
        u1 = itm[1]
        u2 = itm[2]
        pr = itm[3]

        beta = 0.5 * (rho / pr)

        tempq = cuda.local.array((4), dtype=numba.float64)

        tempq[0] = (math.log(rho) + (math.log(beta) * 2.5) - (beta * ((u1*u1) + (u2 * u2))))
        two_times_beta = 2 * beta

        tempq[1] = (two_times_beta * u1)
        tempq[2] = (two_times_beta * u2)
        tempq[3] = -two_times_beta

        q[idx][0] = tempq[0]
        q[idx][1] = tempq[1]
        q[idx][2] = tempq[2]
        q[idx][3] = tempq[3]

@cuda.jit(inline=True)
def q_var_derivatives_cuda_kernel(x, y, maxminq, conn_gpu, nbhs, q, dq, power):
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x
    idx =  bx * bw + tx
    if idx > 0 and idx < len(x):

        x_i = x[idx]
        y_i = y[idx]

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

        # minq = cuda.local.array((4), dtype=numba.float64)
        # maxq = cuda.local.array((4), dtype=numba.float64)

        equalize(maxminq[idx][0], q[idx])
        equalize(maxminq[idx][1], q[idx])
        
        # for i in range(4):
        #     limiters_cuda.minimum(globaldata, idx, i, globaldata[idx]['minq'])
        #     limiters_cuda.maximum(globaldata, idx, i, globaldata[idx]['maxq'])

        for conn in conn_gpu[idx][:nbhs[idx]]:

            for i in range(4):
                if q[conn][i] > maxminq[idx][0][i]:
                    maxminq[idx][0][i] = q[conn][i]
                if q[conn][i] < maxminq[idx][1][i]:
                    maxminq[idx][1][i] = q[conn][i]

            x_k = x[conn]
            y_k = y[conn]

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

            subtract(q[conn], q[idx], temp)
            multiply((weights * delx), temp, temp)
            add(sum_delx_delq, temp, sum_delx_delq)

            temp[0] = 0
            temp[1] = 0
            temp[2] = 0
            temp[3] = 0

            subtract(q[conn], q[idx], temp)
            multiply((weights * dely), temp, temp)
            add(sum_dely_delq, temp, sum_dely_delq)

        # equalize(globaldata[idx]['minq'], minq)
        # equalize(globaldata[idx]['maxq'], maxq)

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
    
        dq[idx][0][0] = tempsumx[0]
        dq[idx][0][1] = tempsumx[1]
        dq[idx][0][2] = tempsumx[2]
        dq[idx][0][3] = tempsumx[3]
        dq[idx][1][0] = tempsumy[0]
        dq[idx][1][1] = tempsumy[1]
        dq[idx][1][2] = tempsumy[2]
        dq[idx][1][3] = tempsumy[3]

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