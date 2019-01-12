# cython: profile=True
# cython: binding=True
# cython: linetrace=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1
import math
import numpy as np
cimport limiters
cimport split_fluxes
cimport quadrant_fluxes
cimport misc
cimport numpy as np

cdef np.ndarray wall_dGx_pos(list globaldata, int idx, dict configData):
    cdef int power = configData["core"]["power"]
    cdef int limiter_flag = configData["core"]["limiter_flag"]

    cdef double sum_delx_sqr = 0
    cdef double sum_dely_sqr = 0
    cdef double sum_delx_dely = 0

    cdef np.ndarray sum_delx_delf = np.zeros((4))
    cdef np.ndarray sum_dely_delf = np.zeros((4))

    cdef double x_i = globaldata[idx].getx()
    cdef double y_i = globaldata[idx].gety()

    cdef float nx = globaldata[idx].getnx()
    cdef float ny = globaldata[idx].getny()
    
    cdef float tx = ny
    cdef float ty = -nx
    cdef long itm
    cdef double x_k, y_k, delx, dely, dels, deln, dist, weights, dels_weights, deln_weights
    cdef np.ndarray qtilde_i
    cdef np.ndarray qtilde_k, G_i, G_k
    cdef np.ndarray phi_i, phi_k
    cdef np.ndarray maxi, mini
    cdef int i
    cdef double det

    for itm in globaldata[idx].get_xpos_conn():
        x_k = globaldata[itm].getx()
        y_k = globaldata[itm].gety()

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

        qtilde_i = globaldata[idx].getq() - 0.5*(delx*globaldata[idx].getdq()[0] + dely*globaldata[idx].getdq()[1])
        qtilde_k = globaldata[itm].getq() - 0.5*(delx*globaldata[itm].getdq()[0] + dely*globaldata[itm].getdq()[1])

        if limiter_flag == 1:
            phi_i = np.array(limiters.venkat_limiter(qtilde_i, globaldata, idx, configData))
            phi_k = np.array(limiters.venkat_limiter(qtilde_k, globaldata, itm, configData))
            qtilde_i = globaldata[idx].getq() - 0.5 * phi_i * (delx*globaldata[idx].getdq()[0] + dely*globaldata[idx].getdq()[1])
            qtilde_k = globaldata[itm].getq() - 0.5 * phi_k * (delx*globaldata[itm].getdq()[0] + dely*globaldata[itm].getdq()[1])

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


        result = misc.qtilde_to_primitive(qtilde_i, configData)
        G_i = quadrant_fluxes.flux_quad_GxII(nx, ny, result[0], result[1], result[2], result[3])

        result = misc.qtilde_to_primitive(qtilde_k, configData)
        G_k = quadrant_fluxes.flux_quad_GxII(nx, ny, result[0], result[1], result[2], result[3])

        sum_delx_delf = sum_delx_delf + (np.array(G_k) - np.array(G_i)) * dels_weights
        sum_dely_delf = sum_dely_delf + (np.array(G_k) - np.array(G_i)) * deln_weights

    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely

    one_by_det = 1 / det

    G = (sum_delx_delf*sum_dely_sqr - sum_dely_delf*sum_delx_dely)*one_by_det

    return G

cdef np.ndarray wall_dGx_neg(list globaldata, int idx, dict configData):

    cdef int power = configData["core"]["power"]
    cdef int limiter_flag = configData["core"]["limiter_flag"]

    cdef double sum_delx_sqr = 0
    cdef double sum_dely_sqr = 0
    cdef double sum_delx_dely = 0

    cdef np.ndarray sum_delx_delf = np.zeros((4))
    cdef np.ndarray sum_dely_delf = np.zeros((4))

    cdef double x_i = globaldata[idx].getx()
    cdef double y_i = globaldata[idx].gety()

    cdef float nx = globaldata[idx].getnx()
    cdef float ny = globaldata[idx].getny()
    
    cdef float tx = ny
    cdef float ty = -nx
    cdef long itm
    cdef double x_k, y_k, delx, dely, dels, deln, dist, weights, dels_weights, deln_weights
    cdef np.ndarray qtilde_i
    cdef np.ndarray qtilde_k, G_i, G_k
    cdef np.ndarray phi_i, phi_k
    cdef np.ndarray maxi, mini
    cdef int i
    cdef double det
    
    for itm in globaldata[idx].get_xpos_conn():

        x_k = globaldata[itm].getx()
        y_k = globaldata[itm].gety()

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

        qtilde_i = np.array(globaldata[idx].getq()) - 0.5*(delx*np.array(globaldata[idx].getdq()[0]) + dely*np.array(globaldata[idx].getdq()[1]))
        qtilde_k = np.array(globaldata[itm].getq()) - 0.5*(delx*np.array(globaldata[itm].getdq()[0]) + dely*np.array(globaldata[itm].getdq()[1]))
        
        if limiter_flag == 1:
            phi_i = np.array(limiters.venkat_limiter(qtilde_i, globaldata, idx, configData))
            phi_k = np.array(limiters.venkat_limiter(qtilde_k, globaldata, itm, configData))
            qtilde_i = np.array(globaldata[idx].getq()) - 0.5 * phi_i * (delx*np.array(globaldata[idx].getdq()[0]) + dely*np.array(globaldata[idx].getdq()[1]))
            qtilde_k = np.array(globaldata[itm].getq()) - 0.5 * phi_k * (delx*np.array(globaldata[itm].getdq()[0]) + dely*np.array(globaldata[itm].getdq()[1]))
            
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

        result = misc.qtilde_to_primitive(qtilde_i, configData)
        G_i = quadrant_fluxes.flux_quad_GxI(nx, ny, result[0], result[1], result[2], result[3])

        result = misc.qtilde_to_primitive(qtilde_k, configData)
        G_k = quadrant_fluxes.flux_quad_GxI(nx, ny, result[0], result[1], result[2], result[3])

        sum_delx_delf = sum_delx_delf + (np.array(G_k) - np.array(G_i)) * dels_weights
        sum_dely_delf = sum_dely_delf + (np.array(G_k) - np.array(G_i)) * deln_weights


    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1 / det

    G = (sum_delx_delf*sum_dely_sqr - sum_dely_delf*sum_delx_dely)*one_by_det

    return G

cdef np.ndarray wall_dGy_neg(list globaldata, int idx, dict configData):

    cdef int power = configData["core"]["power"]
    cdef int limiter_flag = configData["core"]["limiter_flag"]

    cdef double sum_delx_sqr = 0
    cdef double sum_dely_sqr = 0
    cdef double sum_delx_dely = 0

    cdef np.ndarray sum_delx_delf = np.zeros((4))
    cdef np.ndarray sum_dely_delf = np.zeros((4))

    cdef double x_i = globaldata[idx].getx()
    cdef double y_i = globaldata[idx].gety()

    cdef float nx = globaldata[idx].getnx()
    cdef float ny = globaldata[idx].getny()
    
    cdef float tx = ny
    cdef float ty = -nx
    cdef long itm
    cdef double x_k, y_k, delx, dely, dels, deln, dist, weights, dels_weights, deln_weights
    cdef np.ndarray qtilde_i
    cdef np.ndarray qtilde_k, G_i, G_k
    cdef np.ndarray phi_i, phi_k
    cdef np.ndarray maxi, mini
    cdef int i
    cdef double det

    for itm in globaldata[idx].get_xpos_conn():

        x_k = globaldata[itm].getx()
        y_k = globaldata[itm].gety()

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

        qtilde_i = np.array(globaldata[idx].getq()) - 0.5*(delx*np.array(globaldata[idx].getdq()[0]) + dely*np.array(globaldata[idx].getdq()[1]))
        qtilde_k = np.array(globaldata[itm].getq()) - 0.5*(delx*np.array(globaldata[itm].getdq()[0]) + dely*np.array(globaldata[itm].getdq()[1]))
        
        if limiter_flag == 1:
            phi_i = np.array(limiters.venkat_limiter(qtilde_i, globaldata, idx, configData))
            phi_k = np.array(limiters.venkat_limiter(qtilde_k, globaldata, itm, configData))
            qtilde_i = np.array(globaldata[idx].getq()) - 0.5 * phi_i * (delx*np.array(globaldata[idx].getdq()[0]) + dely*np.array(globaldata[idx].getdq()[1]))
            qtilde_k = np.array(globaldata[itm].getq()) - 0.5 * phi_k * (delx*np.array(globaldata[itm].getdq()[0]) + dely*np.array(globaldata[itm].getdq()[1]))
            
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

        result = misc.qtilde_to_primitive(qtilde_i, configData)
        G_i = split_fluxes.flux_Gyn(nx, ny, result[0], result[1], result[2], result[3])
        
        result = misc.qtilde_to_primitive(qtilde_k, configData)
        G_k = split_fluxes.flux_Gyn(nx, ny, result[0], result[1], result[2], result[3])

        sum_delx_delf = sum_delx_delf + (np.array(G_k) - np.array(G_i)) * dels_weights
        sum_dely_delf = sum_dely_delf + (np.array(G_k) - np.array(G_i)) * deln_weights


    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1 / det

    G = (sum_dely_delf*sum_delx_sqr - sum_delx_delf*sum_delx_dely)*one_by_det

    return G