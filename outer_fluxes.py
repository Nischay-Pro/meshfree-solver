import math
import numpy as np
import limiters
import split_fluxes
import quadrant_fluxes
import core

def outer_dGx_pos(globaldata, idx, configData):

    power = configData["core"]["power"]
    limiter_flag = configData["core"]["limiter_flag"]

    sum_delx_sqr = 0
    sum_dely_sqr = 0
    sum_delx_dely = 0

    sum_delx_delf = np.array([0,0,0,0])
    sum_dely_delf = np.array([0,0,0,0])

    x_i = globaldata[idx].x
    y_i = globaldata[idx].y

    nx = globaldata[idx].nx
    ny = globaldata[idx].ny

    tx = ny
    ty = -nx

    for itm in globaldata[idx].xpos_conn:

        x_k = globaldata[itm].x
        y_k = globaldata[itm].y

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

        qtilde_i = np.array(globaldata[idx].q) - 0.5*(delx*np.array(globaldata[idx].dq[0]) + dely*np.array(globaldata[idx].dq[1]))
        qtilde_k = np.array(globaldata[itm].q) - 0.5*(delx*np.array(globaldata[itm].dq[0]) + dely*np.array(globaldata[itm].dq[1]))

        if limiter_flag == 1:
            phi_i = np.array(limiters.venkat_limiter(qtilde_i, globaldata, idx, configData), dtype=np.float64)
            phi_k = np.array(limiters.venkat_limiter(qtilde_k, globaldata, itm, configData), dtype=np.float64)
            qtilde_i = np.array(globaldata[idx].q) - 0.5 * phi_i * (delx*np.array(globaldata[idx].dq[0]) + dely*np.array(globaldata[idx].dq[1]))
            qtilde_k = np.array(globaldata[itm].q) - 0.5 * phi_k * (delx*np.array(globaldata[itm].dq[0]) + dely*np.array(globaldata[itm].dq[1]))

        if limiter_flag == 2:
            maxi = limiters.max_q_values(globaldata, idx)
            mini = limiters.min_q_values(globaldata, idx)

            for i in range(4):
                if qtilde_i[i] > maxi[i]:
                    qtilde_i[i] = maxi[i]
                
                if qtilde_i[i] < mini[i]:
                    qtilde_i[i] = mini[i]
                
                if qtilde_k[i] > maxi[i]:
                    qtilde_k[i] = maxi[i]
                
                if qtilde_k[i] < mini[i]:
                    qtilde_k[i] = mini[i]

        result = core.qtilde_to_primitive(qtilde_i, configData)
        G_i = quadrant_fluxes.flux_quad_GxIII(nx, ny, result[0], result[1], result[2], result[3])

        result = core.qtilde_to_primitive(qtilde_k, configData)
        G_k = quadrant_fluxes.flux_quad_GxIII(nx, ny, result[0], result[1], result[2], result[3])

        sum_delx_delf = sum_delx_delf + (np.array(G_k, dtype=np.float64) - np.array(G_i, dtype=np.float64)) * dels_weights
        sum_dely_delf = sum_dely_delf + (np.array(G_k, dtype=np.float64) - np.array(G_i, dtype=np.float64)) * deln_weights


    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1 / det

    G = (sum_delx_delf*sum_dely_sqr - sum_dely_delf*sum_delx_dely)*one_by_det

    return G

def outer_dGx_neg(globaldata, idx, configData):

    power = configData["core"]["power"]
    limiter_flag = configData["core"]["limiter_flag"]

    sum_delx_sqr = 0
    sum_dely_sqr = 0
    sum_delx_dely = 0

    sum_delx_delf = np.array([0,0,0,0])
    sum_dely_delf = np.array([0,0,0,0])

    x_i = globaldata[idx].x
    y_i = globaldata[idx].y

    nx = globaldata[idx].nx
    ny = globaldata[idx].ny

    tx = ny
    ty = -nx

    for itm in globaldata[idx].xneg_conn:

        x_k = globaldata[itm].x
        y_k = globaldata[itm].y

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

        qtilde_i = np.array(globaldata[idx].q) - 0.5*(delx*np.array(globaldata[idx].dq[0]) + dely*np.array(globaldata[idx].dq[1]))
        qtilde_k = np.array(globaldata[itm].q) - 0.5*(delx*np.array(globaldata[itm].dq[0]) + dely*np.array(globaldata[itm].dq[1]))
        
        if limiter_flag == 1:
            phi_i = np.array(limiters.venkat_limiter(qtilde_i, globaldata, idx, configData), dtype=np.float64)
            phi_k = np.array(limiters.venkat_limiter(qtilde_k, globaldata, itm, configData), dtype=np.float64)
            qtilde_i = np.array(globaldata[idx].q) - 0.5 * phi_i * (delx*np.array(globaldata[idx].dq[0]) + dely*np.array(globaldata[idx].dq[1]))
            qtilde_k = np.array(globaldata[itm].q) - 0.5 * phi_k * (delx*np.array(globaldata[itm].dq[0]) + dely*np.array(globaldata[itm].dq[1]))
           
        if limiter_flag == 2:
            maxi = limiters.max_q_values(globaldata, idx)
            mini = limiters.min_q_values(globaldata, idx)

            for i in range(4):
                if qtilde_i[i] > maxi[i]:
                    qtilde_i[i] = maxi[i]
                
                if qtilde_i[i] < mini[i]:
                    qtilde_i[i] = mini[i]
                
                if qtilde_k[i] > maxi[i]:
                    qtilde_k[i] = maxi[i]
                
                if qtilde_k[i] < mini[i]:
                    qtilde_k[i] = mini[i]

        result = core.qtilde_to_primitive(qtilde_i, configData)
        G_i = quadrant_fluxes.flux_quad_GxIV(nx, ny, result[0], result[1], result[2], result[3])
        
        result = core.qtilde_to_primitive(qtilde_k, configData)
        G_k = quadrant_fluxes.flux_quad_GxIV(nx, ny, result[0], result[1], result[2], result[3])

        sum_delx_delf = sum_delx_delf + (np.array(G_k, dtype=np.float64) - np.array(G_i, dtype=np.float64)) * dels_weights
        sum_dely_delf = sum_dely_delf + (np.array(G_k, dtype=np.float64) - np.array(G_i, dtype=np.float64)) * deln_weights


    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1 / det

    G = (sum_delx_delf*sum_dely_sqr - sum_dely_delf*sum_delx_dely)*one_by_det

    return G

def outer_dGy_pos(globaldata, idx, configData):

    power = configData["core"]["power"]
    limiter_flag = configData["core"]["limiter_flag"]

    sum_delx_sqr = 0
    sum_dely_sqr = 0
    sum_delx_dely = 0

    sum_delx_delf = np.array([0,0,0,0])
    sum_dely_delf = np.array([0,0,0,0])

    x_i = globaldata[idx].x
    y_i = globaldata[idx].y

    nx = globaldata[idx].nx
    ny = globaldata[idx].ny

    tx = ny
    ty = -nx

    for itm in globaldata[idx].ypos_conn:

        x_k = globaldata[itm].x
        y_k = globaldata[itm].y

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

        qtilde_i = np.array(globaldata[idx].q) - 0.5*(delx*np.array(globaldata[idx].dq[0]) + dely*np.array(globaldata[idx].dq[1]))
        qtilde_k = np.array(globaldata[itm].q) - 0.5*(delx*np.array(globaldata[itm].dq[0]) + dely*np.array(globaldata[itm].dq[1]))
        
        if limiter_flag == 1:
            phi_i = np.array(limiters.venkat_limiter(qtilde_i, globaldata, idx, configData), dtype=np.float64)
            phi_k = np.array(limiters.venkat_limiter(qtilde_k, globaldata, itm, configData), dtype=np.float64)
            qtilde_i = np.array(globaldata[idx].q) - 0.5 * phi_i * (delx*np.array(globaldata[idx].dq[0]) + dely*np.array(globaldata[idx].dq[1]))
            qtilde_k = np.array(globaldata[itm].q) - 0.5 * phi_k * (delx*np.array(globaldata[itm].dq[0]) + dely*np.array(globaldata[itm].dq[1]))
    
        if limiter_flag == 2:
            maxi = limiters.max_q_values(globaldata, idx)
            mini = limiters.min_q_values(globaldata, idx)

            for i in range(4):
                if qtilde_i[i] > maxi[i]:
                    qtilde_i[i] = maxi[i]
                
                if qtilde_i[i] < mini[i]:
                    qtilde_i[i] = mini[i]
                
                if qtilde_k[i] > maxi[i]:
                    qtilde_k[i] = maxi[i]
                
                if qtilde_k[i] < mini[i]:
                    qtilde_k[i] = mini[i]

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