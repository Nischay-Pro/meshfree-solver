import config
import math
import numpy as np
import core
from mpi4py import MPI

def func_delta_mpi(globaldata_local, globaldata_ghost, configData):
    cfl = configData["core"]["cfl"]
    for idx in globaldata_local.keys():
        if idx > 0:
            x_i = globaldata_local[idx].x
            y_i = globaldata_local[idx].y
            min_delt = 1
            for itm in globaldata_local[idx].conn:
                try:
                    rho = globaldata_local[itm].prim[0]
                    u1 = globaldata_local[itm].prim[1]
                    u2 = globaldata_local[itm].prim[2]
                    pr = globaldata_local[itm].prim[3]
                    x_k = globaldata_local[itm].x
                    y_k = globaldata_local[itm].y
                except:
                    rho = globaldata_ghost[itm].prim[0]
                    u1 = globaldata_ghost[itm].prim[1]
                    u2 = globaldata_ghost[itm].prim[2]
                    pr = globaldata_ghost[itm].prim[3]
                    x_k = globaldata_ghost[itm].x
                    y_k = globaldata_ghost[itm].y



                dist = (x_k - x_i)*(x_k - x_i) + (y_k - y_i)*(y_k - y_i)
                dist = math.sqrt(dist)

                mod_u = math.sqrt(u1*u1 + u2*u2)

                delta_t = dist/(mod_u + 3*math.sqrt(pr/rho))

                delta_t = cfl*delta_t

                if min_delt > delta_t:
                    min_delt = delta_t

            globaldata_local[idx].delta = min_delt

    return globaldata_local

def state_update_mpi(globaldata_local, wallindices, outerindices, interiorindices, configData, iter, res_old, comm):
    max_res = 0	
    sum_res_sqr = 0
    for itm in wallindices:
        nx = globaldata_local[itm].nx
        ny = globaldata_local[itm].ny
        U = primitive_to_conserved(globaldata_local, itm, nx, ny)
        temp = U[0]
        U = np.array(U) - (globaldata_local[itm].delta * np.array(globaldata_local[itm].flux_res))

        U[2] = 0

        U2_rot = U[1]
        U3_rot = U[2]
        U[1] = U2_rot*ny + U3_rot*nx
        U[2] = U3_rot*ny - U2_rot*nx


        res_sqr = (U[0] - temp)*(U[0] - temp)


        if res_sqr > max_res:
            max_res = res_sqr
            max_res_point = itm

        sum_res_sqr = sum_res_sqr + res_sqr

        tempU = np.zeros(4, dtype=np.float64)
        tempU[0] = U[0]
        temp = 1 / U[0]
        tempU[1] = U[1]*temp
        tempU[2] = U[2]*temp

        tempU[3] = (0.4*U[3]) - ((0.2*temp)*(U[1]*U[1] + U[2]*U[2]))

        globaldata_local[itm].prim = tempU

    for itm in outerindices:
        nx = globaldata_local[itm].nx
        ny = globaldata_local[itm].ny

        U = conserved_vector_Ubar(globaldata_local, itm, nx, ny, configData)

        temp = U[0]

        U = np.array(U) - globaldata_local[itm].delta * np.array(globaldata_local[itm].flux_res)

        U2_rot = U[1]
        U3_rot = U[2]

        U[1] = U2_rot*ny + U3_rot*nx
        U[2] = U3_rot*ny - U2_rot*nx

        tempU = np.zeros(4, dtype=np.float64)
        tempU[0] = U[0]
        temp = 1 / U[0]
        tempU[1] = U[1]*temp
        tempU[2] = U[2]*temp
        tempU[3] = (0.4*U[3]) - (0.2*temp)*(U[1]*U[1] + U[2]*U[2])
        

        globaldata_local[itm].prim = tempU

    for itm in interiorindices:
        nx = globaldata_local[itm].nx
        ny = globaldata_local[itm].ny
        U = primitive_to_conserved(globaldata_local, itm, nx, ny)
        temp = U[0]

        U = np.array(U) - globaldata_local[itm].delta * np.array(globaldata_local[itm].flux_res)
        U2_rot = U[1]
        U3_rot = U[2]
        U[1] = U2_rot*ny + U3_rot*nx
        U[2] = U3_rot*ny - U2_rot*nx

        res_sqr = (U[0] - temp)*(U[0] - temp)

        if res_sqr > max_res:
            max_res = res_sqr
            max_res_point = itm

        sum_res_sqr = sum_res_sqr + res_sqr

        tempU = np.zeros(4, dtype=np.float64)
        tempU[0] = U[0]
        temp = 1 / U[0]
        tempU[1] = U[1]*temp
        tempU[2] = U[2]*temp
        tempU[3] = (0.4*U[3]) - (0.2*temp)*(U[1]*U[1] + U[2]*U[2])
        globaldata_local[itm].prim = tempU

    total_res_sqr = np.zeros(1, dtype="float64")
    total_points = np.zeros(1, dtype="int64")
    comm.Reduce(np.array(sum_res_sqr), total_res_sqr, op=MPI.SUM, root=0)
    comm.Reduce(np.array(len(globaldata_local)), total_points, op=MPI.SUM, root=0)
    if comm.Get_rank() == 0:
        res_new = math.sqrt(total_res_sqr[0])/ total_points[0]
        if iter <= 2:
            res_old = res_new
            residue = 0
        else:
            residue = math.log10(res_new/res_old)

        print(res_old)

        # with open('residue', 'a') as the_file:
        #     the_file.write("%i %f" % (iter, residue))
        
        print("Iteration Number: ", iter)
        print("Residue: ", residue)

    return globaldata_local, res_old

def primitive_to_conserved(globaldata, itm, nx, ny):

    U = []

    rho = globaldata[itm].prim[0]
    U.append(rho) 
    temp1 = rho*globaldata[itm].prim[1]
    temp2 = rho*globaldata[itm].prim[2]

    U.append(temp1*ny - temp2*nx)
    U.append(temp1*nx + temp2*ny)
    U.append(2.5*globaldata[itm].prim[3] + 0.5*(temp1*temp1 + temp2*temp2)/rho)

    return U

def conserved_vector_Ubar(globaldata, itm, nx, ny, configData):
    Mach = configData["core"]["mach"]
    gamma = configData["core"]["gamma"]
    pr_inf = configData["core"]["pr_inf"]
    rho_inf = configData["core"]["rho_inf"]

    Ubar = []

    theta = core.calculateTheta(configData)

    u1_inf = Mach*math.cos(theta)
    u2_inf = Mach*math.sin(theta)

    tx = ny
    ty = -nx

    u1_inf_rot = u1_inf*tx + u2_inf*ty
    u2_inf_rot = u1_inf*nx + u2_inf*ny

    temp1 = (u1_inf_rot*u1_inf_rot + u2_inf_rot*u2_inf_rot)
    e_inf = pr_inf/(rho_inf*(gamma-1)) + 0.5*(temp1)

    beta = (0.5*rho_inf)/pr_inf
    S2 = u2_inf_rot*math.sqrt(beta)
    B2_inf = math.exp(-S2*S2)/(2*math.sqrt(math.pi*beta))
    A2n_inf = 0.5*(1-math.erf(S2))

    rho = globaldata[itm].prim[0]
    u1 = globaldata[itm].prim[1]
    u2 = globaldata[itm].prim[2]
    pr = globaldata[itm].prim[3]

    u1_rot = u1*tx + u2*ty
    u2_rot = u1*nx + u2*ny

    temp1 = (u1_rot*u1_rot + u2_rot*u2_rot)
    e = pr/(rho*(gamma-1)) + 0.5*(temp1)

    beta = (rho)/(2*pr)
    S2 = u2_rot*math.sqrt(beta)
    B2 = math.exp(-S2*S2)/(2*math.sqrt(math.pi*beta))
    A2p = 0.5*(1+math.erf(S2))

    Ubar.append((rho_inf*A2n_inf) + (rho*A2p))

    Ubar.append((rho_inf*u1_inf_rot*A2n_inf) + (rho*u1_rot*A2p))

    temp1 = rho_inf*(u2_inf_rot*A2n_inf - B2_inf)
    temp2 = rho*(u2_rot*A2p + B2)
    Ubar.append(temp1 + temp2)

    temp1 = (rho_inf*A2n_inf*e_inf - 0.5*rho_inf*u2_inf_rot*B2_inf)
    temp2 = (rho*A2p*e + 0.5*rho*u2_rot*B2)

    Ubar.append(temp1 + temp2)

    return Ubar