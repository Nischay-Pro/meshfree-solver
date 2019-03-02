import math
import numpy as np
import limiters_mpi
import split_fluxes
import quadrant_fluxes
import core

def interior_dGx_pos(globaldata_local, globaldata_ghost, idx, configData):

    power = configData["core"]["power"]
    limiter_flag = configData["core"]["limiter_flag"]

    sum_delx_sqr = 0
    sum_dely_sqr = 0
    sum_delx_dely = 0

    sum_delx_delf = np.array([0,0,0,0])
    sum_dely_delf = np.array([0,0,0,0])

    x_i = globaldata_local[idx].x
    y_i = globaldata_local[idx].y

    nx = globaldata_local[idx].nx
    ny = globaldata_local[idx].ny

    tx = ny
    ty = -nx

    for itm in globaldata_local[idx].xpos_conn:
        ghost = False
        try:
            x_k = globaldata_local[itm].x
            y_k = globaldata_local[itm].y
        except:
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

        qtilde_i = np.array(globaldata_local[idx].q) - 0.5*(delx*np.array(globaldata_local[idx].dq[0]) + dely*np.array(globaldata_local[idx].dq[1]))
        if not ghost:
            qtilde_k = np.array(globaldata_local[itm].q) - 0.5*(delx*np.array(globaldata_local[itm].dq[0]) + dely*np.array(globaldata_local[itm].dq[1]))
        else:
            qtilde_k = np.array(globaldata_ghost[itm].q) - 0.5*(delx*np.array(globaldata_ghost[itm].dq[0]) + dely*np.array(globaldata_ghost[itm].dq[1]))

        if limiter_flag == 1:

            phi_i = np.array(limiters_mpi.venkat_limiter(qtilde_i, globaldata_local, globaldata_ghost, idx, configData), dtype=np.float64)
            phi_k = np.array(limiters_mpi.venkat_limiter(qtilde_k, globaldata_local, globaldata_ghost, itm, configData), dtype=np.float64)
            
            qtilde_i = np.array(globaldata_local[idx].q) - 0.5 * phi_i * (delx*np.array(globaldata_local[idx].dq[0]) + dely*np.array(globaldata_local[idx].dq[1]))
            if not ghost:
                qtilde_k = np.array(globaldata_local[itm].q) - 0.5 * phi_k * (delx*np.array(globaldata_local[itm].dq[0]) + dely*np.array(globaldata_local[itm].dq[1]))
            else:
                qtilde_k = np.array(globaldata_ghost[itm].q) - 0.5 * phi_k * (delx*np.array(globaldata_ghost[itm].dq[0]) + dely*np.array(globaldata_ghost[itm].dq[1]))


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

    sum_delx_delf = np.array([0,0,0,0])
    sum_dely_delf = np.array([0,0,0,0])

    x_i = globaldata_local[idx].x
    y_i = globaldata_local[idx].y

    nx = globaldata_local[idx].nx
    ny = globaldata_local[idx].ny

    tx = ny
    ty = -nx

    for itm in globaldata_local[idx].xneg_conn:
        ghost = False
        try:
            x_k = globaldata_local[itm].x
            y_k = globaldata_local[itm].y
        except:
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

        qtilde_i = np.array(globaldata_local[idx].q) - 0.5*(delx*np.array(globaldata_local[idx].dq[0]) + dely*np.array(globaldata_local[idx].dq[1]))
        if not ghost:
            qtilde_k = np.array(globaldata_local[itm].q) - 0.5*(delx*np.array(globaldata_local[itm].dq[0]) + dely*np.array(globaldata_local[itm].dq[1]))
        else:
            qtilde_k = np.array(globaldata_ghost[itm].q) - 0.5*(delx*np.array(globaldata_ghost[itm].dq[0]) + dely*np.array(globaldata_ghost[itm].dq[1]))

        if limiter_flag == 1:

            phi_i = np.array(limiters_mpi.venkat_limiter(qtilde_i, globaldata_local, globaldata_ghost, idx, configData), dtype=np.float64)
            phi_k = np.array(limiters_mpi.venkat_limiter(qtilde_k, globaldata_local, globaldata_ghost, itm, configData), dtype=np.float64)
            
            qtilde_i = np.array(globaldata_local[idx].q) - 0.5 * phi_i * (delx*np.array(globaldata_local[idx].dq[0]) + dely*np.array(globaldata_local[idx].dq[1]))
            if not ghost:
                qtilde_k = np.array(globaldata_local[itm].q) - 0.5 * phi_k * (delx*np.array(globaldata_local[itm].dq[0]) + dely*np.array(globaldata_local[itm].dq[1]))
            else:
                qtilde_k = np.array(globaldata_ghost[itm].q) - 0.5 * phi_k * (delx*np.array(globaldata_ghost[itm].dq[0]) + dely*np.array(globaldata_ghost[itm].dq[1]))

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

    sum_delx_delf = np.array([0,0,0,0])
    sum_dely_delf = np.array([0,0,0,0])

    x_i = globaldata_local[idx].x
    y_i = globaldata_local[idx].y

    nx = globaldata_local[idx].nx
    ny = globaldata_local[idx].ny

    tx = ny
    ty = -nx

    for itm in globaldata_local[idx].ypos_conn:
        ghost = False
        try:
            x_k = globaldata_local[itm].x
            y_k = globaldata_local[itm].y
        except:
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

        qtilde_i = np.array(globaldata_local[idx].q) - 0.5*(delx*np.array(globaldata_local[idx].dq[0]) + dely*np.array(globaldata_local[idx].dq[1]))
        if not ghost:
            qtilde_k = np.array(globaldata_local[itm].q) - 0.5*(delx*np.array(globaldata_local[itm].dq[0]) + dely*np.array(globaldata_local[itm].dq[1]))
        else:
            qtilde_k = np.array(globaldata_ghost[itm].q) - 0.5*(delx*np.array(globaldata_ghost[itm].dq[0]) + dely*np.array(globaldata_ghost[itm].dq[1]))

        if limiter_flag == 1:

            phi_i = np.array(limiters_mpi.venkat_limiter(qtilde_i, globaldata_local, globaldata_ghost, idx, configData), dtype=np.float64)
            phi_k = np.array(limiters_mpi.venkat_limiter(qtilde_k, globaldata_local, globaldata_ghost, itm, configData), dtype=np.float64)
            
            qtilde_i = np.array(globaldata_local[idx].q) - 0.5 * phi_i * (delx*np.array(globaldata_local[idx].dq[0]) + dely*np.array(globaldata_local[idx].dq[1]))
            if not ghost:
                qtilde_k = np.array(globaldata_local[itm].q) - 0.5 * phi_k * (delx*np.array(globaldata_local[itm].dq[0]) + dely*np.array(globaldata_local[itm].dq[1]))
            else:
                qtilde_k = np.array(globaldata_ghost[itm].q) - 0.5 * phi_k * (delx*np.array(globaldata_ghost[itm].dq[0]) + dely*np.array(globaldata_ghost[itm].dq[1]))


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

    sum_delx_delf = np.array([0,0,0,0])
    sum_dely_delf = np.array([0,0,0,0])

    x_i = globaldata_local[idx].x
    y_i = globaldata_local[idx].y

    nx = globaldata_local[idx].nx
    ny = globaldata_local[idx].ny

    tx = ny
    ty = -nx

    for itm in globaldata_local[idx].yneg_conn:
        ghost = False
        try:
            x_k = globaldata_local[itm].x
            y_k = globaldata_local[itm].y
        except:
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

        qtilde_i = np.array(globaldata_local[idx].q) - 0.5*(delx*np.array(globaldata_local[idx].dq[0]) + dely*np.array(globaldata_local[idx].dq[1]))
        if not ghost:
            qtilde_k = np.array(globaldata_local[itm].q) - 0.5*(delx*np.array(globaldata_local[itm].dq[0]) + dely*np.array(globaldata_local[itm].dq[1]))
        else:
            qtilde_k = np.array(globaldata_ghost[itm].q) - 0.5*(delx*np.array(globaldata_ghost[itm].dq[0]) + dely*np.array(globaldata_ghost[itm].dq[1]))

        if limiter_flag == 1:

            phi_i = np.array(limiters_mpi.venkat_limiter(qtilde_i, globaldata_local, globaldata_ghost, idx, configData), dtype=np.float64)
            phi_k = np.array(limiters_mpi.venkat_limiter(qtilde_k, globaldata_local, globaldata_ghost, itm, configData), dtype=np.float64)
            
            qtilde_i = np.array(globaldata_local[idx].q) - 0.5 * phi_i * (delx*np.array(globaldata_local[idx].dq[0]) + dely*np.array(globaldata_local[idx].dq[1]))
            if not ghost:
                qtilde_k = np.array(globaldata_local[itm].q) - 0.5 * phi_k * (delx*np.array(globaldata_local[itm].dq[0]) + dely*np.array(globaldata_local[itm].dq[1]))
            else:
                qtilde_k = np.array(globaldata_ghost[itm].q) - 0.5 * phi_k * (delx*np.array(globaldata_ghost[itm].dq[0]) + dely*np.array(globaldata_ghost[itm].dq[1]))


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