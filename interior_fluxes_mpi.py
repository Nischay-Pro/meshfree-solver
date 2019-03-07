import math
import numpy as np
import limiters_mpi
import split_fluxes
import quadrant_fluxes
import core
from numba import jit

def interior_dGx_pos(globaldata_local, globaldata_ghost, idx, configData):

    power = configData["core"]["power"]
    limiter_flag = configData["core"]["limiter_flag"]

    sum_delx_sqr = 0
    sum_dely_sqr = 0
    sum_delx_dely = 0

    sum_delx_delf = np.zeros(4, dtype=np.float64)
    sum_dely_delf = np.zeros(4, dtype=np.float64)

    x_i = globaldata_local[idx].x
    y_i = globaldata_local[idx].y

    nx = globaldata_local[idx].nx
    ny = globaldata_local[idx].ny

    tx = ny
    ty = -nx

    for itm in globaldata_local[idx].xpos_conn:
        ghost = False
        if itm in globaldata_local:
            x_k = globaldata_local[itm].x
            y_k = globaldata_local[itm].y
        else:
            x_k = globaldata_ghost[itm].x
            y_k = globaldata_ghost[itm].y
            ghost = True

        delx = x_k - x_i
        dely = y_k - y_i

        dels = delx*tx + dely*ty
        deln = delx*nx + dely*ny

        dist = math.sqrt(dels*dels + deln*deln)
        weights = dist**power

        dels_weights = dels*weights
        deln_weights = deln*weights

        sum_delx_sqr = sum_delx_sqr + dels*dels_weights
        sum_dely_sqr = sum_dely_sqr + deln*deln_weights

        sum_delx_dely = sum_delx_dely + dels*deln_weights

        qtilde_i = globaldata_local[idx].q - (0.5*(delx*globaldata_local[idx].dq[0] + dely*globaldata_local[idx].dq[1]))
        if not ghost:
            qtilde_k = globaldata_local[itm].q - (0.5*(delx*globaldata_local[itm].dq[0] + dely*globaldata_local[itm].dq[1]))
        else:
            qtilde_k = globaldata_ghost[itm].q - (0.5*(delx*globaldata_ghost[itm].dq[0] + dely*globaldata_ghost[itm].dq[1]))

        if limiter_flag == 1:

            phi_i = limiters_mpi.venkat_limiter(qtilde_i, globaldata_local, globaldata_ghost, idx, configData)
            phi_k = limiters_mpi.venkat_limiter(qtilde_k, globaldata_local, globaldata_ghost, itm, configData)
            
            qtilde_i = globaldata_local[idx].q - (0.5 * phi_i * (delx*globaldata_local[idx].dq[0] + dely*globaldata_local[idx].dq[1]))
            if not ghost:
                qtilde_k = globaldata_local[itm].q - (0.5 * phi_k * (delx*globaldata_local[itm].dq[0] + dely*globaldata_local[itm].dq[1]))
            else:
                qtilde_k = globaldata_ghost[itm].q - (0.5 * phi_k * (delx*globaldata_ghost[itm].dq[0] + dely*globaldata_ghost[itm].dq[1]))

        result = core.qtilde_to_primitive(qtilde_i, configData)
        G_i = split_fluxes.flux_Gxp(nx, ny, result[0], result[1], result[2], result[3])

        result = core.qtilde_to_primitive(qtilde_k, configData)
        G_k = split_fluxes.flux_Gxp(nx, ny, result[0], result[1], result[2], result[3])

        sum_delx_delf = sum_delx_delf + (np.array(G_k, dtype=np.float64) - np.array(G_i, dtype=np.float64)) * dels_weights
        sum_dely_delf = sum_dely_delf + (np.array(G_k, dtype=np.float64) - np.array(G_i, dtype=np.float64)) * deln_weights


    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1 / det

    G = (sum_delx_delf*sum_dely_sqr - sum_dely_delf*sum_delx_dely)*one_by_det

    return G

def interior_dGx_neg(globaldata_local, globaldata_ghost, idx, configData):

    power = configData["core"]["power"]
    limiter_flag = configData["core"]["limiter_flag"]

    sum_delx_sqr = 0
    sum_dely_sqr = 0
    sum_delx_dely = 0

    sum_delx_delf = np.zeros(4, dtype=np.float64)
    sum_dely_delf = np.zeros(4, dtype=np.float64)

    x_i = globaldata_local[idx].x
    y_i = globaldata_local[idx].y

    nx = globaldata_local[idx].nx
    ny = globaldata_local[idx].ny

    tx = ny
    ty = -nx

    for itm in globaldata_local[idx].xneg_conn:
        ghost = False
        if itm in globaldata_local:
            x_k = globaldata_local[itm].x
            y_k = globaldata_local[itm].y
        else:
            x_k = globaldata_ghost[itm].x
            y_k = globaldata_ghost[itm].y
            ghost = True

        delx = x_k - x_i
        dely = y_k - y_i

        dels = delx*tx + dely*ty
        deln = delx*nx + dely*ny

        dist = math.sqrt(dels*dels + deln*deln)
        weights = dist**power

        dels_weights = dels*weights
        deln_weights = deln*weights

        sum_delx_sqr = sum_delx_sqr + dels*dels_weights
        sum_dely_sqr = sum_dely_sqr + deln*deln_weights

        sum_delx_dely = sum_delx_dely + dels*deln_weights

        qtilde_i = globaldata_local[idx].q - (0.5*(delx*globaldata_local[idx].dq[0] + dely*globaldata_local[idx].dq[1]))
        if not ghost:
            qtilde_k = globaldata_local[itm].q - (0.5*(delx*globaldata_local[itm].dq[0] + dely*globaldata_local[itm].dq[1]))
        else:
            qtilde_k = globaldata_ghost[itm].q - (0.5*(delx*globaldata_ghost[itm].dq[0] + dely*globaldata_ghost[itm].dq[1]))

        if limiter_flag == 1:

            phi_i = limiters_mpi.venkat_limiter(qtilde_i, globaldata_local, globaldata_ghost, idx, configData)
            phi_k = limiters_mpi.venkat_limiter(qtilde_k, globaldata_local, globaldata_ghost, itm, configData)
            
            qtilde_i = globaldata_local[idx].q - (0.5 * phi_i * (delx*globaldata_local[idx].dq[0] + dely*globaldata_local[idx].dq[1]))
            if not ghost:
                qtilde_k = globaldata_local[itm].q - (0.5 * phi_k * (delx*globaldata_local[itm].dq[0] + dely*globaldata_local[itm].dq[1]))
            else:
                qtilde_k = globaldata_ghost[itm].q - (0.5 * phi_k * (delx*globaldata_ghost[itm].dq[0] + dely*globaldata_ghost[itm].dq[1]))
                
        result = core.qtilde_to_primitive(qtilde_i, configData)
        G_i = split_fluxes.flux_Gxn(nx, ny, result[0], result[1], result[2], result[3])

        result = core.qtilde_to_primitive(qtilde_k, configData)
        G_k = split_fluxes.flux_Gxn(nx, ny, result[0], result[1], result[2], result[3])

        sum_delx_delf = sum_delx_delf + (np.array(G_k, dtype=np.float64) - np.array(G_i, dtype=np.float64)) * dels_weights
        sum_dely_delf = sum_dely_delf + (np.array(G_k, dtype=np.float64) - np.array(G_i, dtype=np.float64)) * deln_weights


    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1 / det

    G = (sum_delx_delf*sum_dely_sqr - sum_dely_delf*sum_delx_dely)*one_by_det

    return G

def interior_dGy_pos(globaldata_local, globaldata_ghost, idx, configData):

    power = configData["core"]["power"]
    limiter_flag = configData["core"]["limiter_flag"]

    sum_delx_sqr = 0
    sum_dely_sqr = 0
    sum_delx_dely = 0

    sum_delx_delf = np.zeros(4, dtype=np.float64)
    sum_dely_delf = np.zeros(4, dtype=np.float64)

    x_i = globaldata_local[idx].x
    y_i = globaldata_local[idx].y

    nx = globaldata_local[idx].nx
    ny = globaldata_local[idx].ny

    tx = ny
    ty = -nx

    for itm in globaldata_local[idx].ypos_conn:
        ghost = False
        if itm in globaldata_local:
            x_k = globaldata_local[itm].x
            y_k = globaldata_local[itm].y
        else:
            x_k = globaldata_ghost[itm].x
            y_k = globaldata_ghost[itm].y
            ghost = True

        delx = x_k - x_i
        dely = y_k - y_i

        dels = delx*tx + dely*ty
        deln = delx*nx + dely*ny

        dist = math.sqrt(dels*dels + deln*deln)
        weights = dist**power

        dels_weights = dels*weights
        deln_weights = deln*weights

        sum_delx_sqr = sum_delx_sqr + dels*dels_weights
        sum_dely_sqr = sum_dely_sqr + deln*deln_weights

        sum_delx_dely = sum_delx_dely + dels*deln_weights

        qtilde_i = globaldata_local[idx].q - (0.5*(delx*globaldata_local[idx].dq[0] + dely*globaldata_local[idx].dq[1]))
        if not ghost:
            qtilde_k = globaldata_local[itm].q - (0.5*(delx*globaldata_local[itm].dq[0] + dely*globaldata_local[itm].dq[1]))
        else:
            qtilde_k = globaldata_ghost[itm].q - (0.5*(delx*globaldata_ghost[itm].dq[0] + dely*globaldata_ghost[itm].dq[1]))

        if limiter_flag == 1:

            phi_i = limiters_mpi.venkat_limiter(qtilde_i, globaldata_local, globaldata_ghost, idx, configData)
            phi_k = limiters_mpi.venkat_limiter(qtilde_k, globaldata_local, globaldata_ghost, itm, configData)
            
            qtilde_i = globaldata_local[idx].q - (0.5 * phi_i * (delx*globaldata_local[idx].dq[0] + dely*globaldata_local[idx].dq[1]))
            if not ghost:
                qtilde_k = globaldata_local[itm].q - (0.5 * phi_k * (delx*globaldata_local[itm].dq[0] + dely*globaldata_local[itm].dq[1]))
            else:
                qtilde_k = globaldata_ghost[itm].q - (0.5 * phi_k * (delx*globaldata_ghost[itm].dq[0] + dely*globaldata_ghost[itm].dq[1]))
        result = core.qtilde_to_primitive(qtilde_i, configData)
        G_i = split_fluxes.flux_Gyp(nx, ny, result[0], result[1], result[2], result[3])
        
        result = core.qtilde_to_primitive(qtilde_k, configData)
        G_k = split_fluxes.flux_Gyp(nx, ny, result[0], result[1], result[2], result[3])

        sum_delx_delf = sum_delx_delf + (np.array(G_k, dtype=np.float64) - np.array(G_i, dtype=np.float64)) * dels_weights
        sum_dely_delf = sum_dely_delf + (np.array(G_k, dtype=np.float64) - np.array(G_i, dtype=np.float64)) * deln_weights


    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1 / det

    G = (sum_dely_delf*sum_delx_sqr - sum_delx_delf*sum_delx_dely)*one_by_det

    return G

def interior_dGy_neg(globaldata_local, globaldata_ghost, idx, configData):

    power = configData["core"]["power"]
    limiter_flag = configData["core"]["limiter_flag"]

    sum_delx_sqr = 0
    sum_dely_sqr = 0
    sum_delx_dely = 0

    sum_delx_delf = np.zeros(4, dtype=np.float64)
    sum_dely_delf = np.zeros(4, dtype=np.float64)

    x_i = globaldata_local[idx].x
    y_i = globaldata_local[idx].y

    nx = globaldata_local[idx].nx
    ny = globaldata_local[idx].ny

    tx = ny
    ty = -nx

    for itm in globaldata_local[idx].yneg_conn:
        ghost = False
        if itm in globaldata_local:
            x_k = globaldata_local[itm].x
            y_k = globaldata_local[itm].y
        else:
            x_k = globaldata_ghost[itm].x
            y_k = globaldata_ghost[itm].y
            ghost = True

        delx = x_k - x_i
        dely = y_k - y_i

        dels = delx*tx + dely*ty
        deln = delx*nx + dely*ny

        dist = math.sqrt(dels*dels + deln*deln)
        weights = dist**power

        dels_weights = dels*weights
        deln_weights = deln*weights

        sum_delx_sqr = sum_delx_sqr + dels*dels_weights
        sum_dely_sqr = sum_dely_sqr + deln*deln_weights

        sum_delx_dely = sum_delx_dely + dels*deln_weights

        qtilde_i = globaldata_local[idx].q - (0.5*(delx*globaldata_local[idx].dq[0] + dely*globaldata_local[idx].dq[1]))
        if not ghost:
            qtilde_k = globaldata_local[itm].q - (0.5*(delx*globaldata_local[itm].dq[0] + dely*globaldata_local[itm].dq[1]))
        else:
            qtilde_k = globaldata_ghost[itm].q - (0.5*(delx*globaldata_ghost[itm].dq[0] + dely*globaldata_ghost[itm].dq[1]))

        if limiter_flag == 1:

            phi_i = limiters_mpi.venkat_limiter(qtilde_i, globaldata_local, globaldata_ghost, idx, configData)
            phi_k = limiters_mpi.venkat_limiter(qtilde_k, globaldata_local, globaldata_ghost, itm, configData)
            
            qtilde_i = globaldata_local[idx].q - (0.5 * phi_i * (delx*globaldata_local[idx].dq[0] + dely*globaldata_local[idx].dq[1]))
            if not ghost:
                qtilde_k = globaldata_local[itm].q - (0.5 * phi_k * (delx*globaldata_local[itm].dq[0] + dely*globaldata_local[itm].dq[1]))
            else:
                qtilde_k = globaldata_ghost[itm].q - (0.5 * phi_k * (delx*globaldata_ghost[itm].dq[0] + dely*globaldata_ghost[itm].dq[1]))


        result = core.qtilde_to_primitive(qtilde_i, configData)
        G_i = split_fluxes.flux_Gyn(nx, ny, result[0], result[1], result[2], result[3])
        
        result = core.qtilde_to_primitive(qtilde_k, configData)
        G_k = split_fluxes.flux_Gyn(nx, ny, result[0], result[1], result[2], result[3])

        sum_delx_delf = sum_delx_delf + (np.array(G_k, dtype=np.float64) - np.array(G_i, dtype=np.float64)) * dels_weights
        sum_dely_delf = sum_dely_delf + (np.array(G_k, dtype=np.float64) - np.array(G_i, dtype=np.float64)) * deln_weights


    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1 / det

    G = (sum_dely_delf*sum_delx_sqr - sum_delx_delf*sum_delx_dely)*one_by_det

    return G