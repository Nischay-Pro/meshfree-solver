import math
import limiters_cuda
import split_fluxes_cuda
import quadrant_fluxes_cuda
import numba
from numba import cuda
from cuda_func import add, zeros, multiply, qtilde_to_primitive_cuda, subtract, multiply_element_wise

@cuda.jit(device=True)
def outer_dGx_pos(globaldata, idx, power, vl_const, gamma, store):

    sum_delx_sqr = 0
    sum_dely_sqr = 0
    sum_delx_dely = 0

    sum_delx_delf = cuda.local.array((4), numba.float64)
    sum_dely_delf = cuda.local.array((4), numba.float64)

    qtilde_i = cuda.local.array((4), numba.float64)
    qtilde_k = cuda.local.array((4), numba.float64)

    phi_i = cuda.local.array((4), numba.float64)
    phi_k = cuda.local.array((4), numba.float64)

    temp1 = cuda.local.array((4), numba.float64)
    temp2 = cuda.local.array((4), numba.float64)

    result = cuda.local.array((4), numba.float64)
    G_i = cuda.local.array((4), numba.float64)
    G_k = cuda.local.array((4), numba.float64)



    zeros(sum_delx_delf, sum_delx_delf)
    zeros(sum_dely_delf, sum_dely_delf)

    x_i = globaldata[idx]['x']
    y_i = globaldata[idx]['y']

    nx = globaldata[idx]['nx']
    ny = globaldata[idx]['ny']

    tx = ny
    ty = -nx

    for itm in globaldata[idx]['xpos_conn'][:globaldata[idx]['xpos_nbhs']]:

        x_k = globaldata[itm]['x']
        y_k = globaldata[itm]['y']

        delx = x_k - x_i
        dely = y_k - y_i

        dels = delx*tx + dely*ty
        deln = delx*nx + dely*ny

        dist = (dels*dels + deln*deln) ** 0.5
        weights = dist**power

        dels_weights = dels*weights
        deln_weights = deln*weights

        sum_delx_sqr = sum_delx_sqr + dels*dels_weights
        sum_dely_sqr = sum_dely_sqr + deln*deln_weights

        sum_delx_dely = sum_delx_dely + dels*deln_weights

        zeros(qtilde_i, qtilde_i)
        zeros(qtilde_k, qtilde_k)

        zeros(temp1, temp1)
        zeros(temp2, temp2)

        multiply(delx, globaldata[idx]['dq'][0], temp1)
        multiply(dely, globaldata[idx]['dq'][1], temp2)

        add(temp1, temp2, temp1)
        multiply(0.5, temp1, temp1)
        subtract(globaldata[idx]['q'], temp1, qtilde_i)

        zeros(temp1, temp1)
        zeros(temp2, temp2)

        multiply(delx, globaldata[itm]['dq'][0], temp1)
        multiply(dely, globaldata[itm]['dq'][1], temp2)

        add(temp1, temp2, temp1)
        multiply(0.5, temp1, temp1)
        subtract(globaldata[itm]['q'], temp1, qtilde_k)

        zeros(phi_i, phi_i)
        zeros(phi_k, phi_k)

        limiters_cuda.venkat_limiter(qtilde_i, globaldata, idx, vl_const, phi_i)

        limiters_cuda.venkat_limiter(qtilde_k, globaldata, itm, vl_const, phi_k)

        zeros(temp1, temp1)
        zeros(temp2, temp2)

        multiply(delx, globaldata[idx]['dq'][0], temp1)
        multiply(dely, globaldata[idx]['dq'][1], temp2)

        multiply(0.5, phi_i, phi_i)

        add(temp1, temp2, temp1)
        multiply_element_wise(temp1, phi_i, temp1)

        subtract(globaldata[idx]['q'], temp1, qtilde_i)

        zeros(temp1, temp1)
        zeros(temp2, temp2)

        multiply(delx, globaldata[itm]['dq'][0], temp1)
        multiply(dely, globaldata[itm]['dq'][1], temp2)

        multiply(0.5, phi_k, phi_k)

        add(temp1, temp2, temp1)
        multiply_element_wise(temp1, phi_k, temp1)

        subtract(globaldata[itm]['q'], temp1, qtilde_k)

        zeros(result, result)

        qtilde_to_primitive_cuda(qtilde_i, gamma, result)

        quadrant_fluxes_cuda.flux_quad_GxIII(nx, ny, result[0], result[1], result[2], result[3], G_i)

        qtilde_to_primitive_cuda(qtilde_k, gamma, result)

        quadrant_fluxes_cuda.flux_quad_GxIII(nx, ny, result[0], result[1], result[2], result[3], G_k)

        zeros(temp1, temp1)
        subtract(G_k, G_i, temp1)
        multiply(dels_weights, temp1, temp1)
        add(sum_delx_delf, temp1, sum_delx_delf)

        zeros(temp2, temp2)
        subtract(G_k, G_i, temp2)
        multiply(deln_weights, temp2, temp2)
        add(sum_dely_delf, temp2, sum_dely_delf)

    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1 / det

    zeros(store, store)
    zeros(temp1, temp1)

    multiply(sum_dely_sqr, sum_delx_delf, sum_delx_delf)
    multiply(sum_delx_dely, sum_dely_delf, sum_dely_delf)

    subtract(sum_delx_delf, sum_dely_delf, temp1)

    multiply(one_by_det, temp1, store)

@cuda.jit(device=True)
def outer_dGx_neg(globaldata, idx, power, vl_const, gamma, store):

    sum_delx_sqr = 0
    sum_dely_sqr = 0
    sum_delx_dely = 0

    sum_delx_delf = cuda.local.array((4), numba.float64)
    sum_dely_delf = cuda.local.array((4), numba.float64)

    qtilde_i = cuda.local.array((4), numba.float64)
    qtilde_k = cuda.local.array((4), numba.float64)

    phi_i = cuda.local.array((4), numba.float64)
    phi_k = cuda.local.array((4), numba.float64)

    temp1 = cuda.local.array((4), numba.float64)
    temp2 = cuda.local.array((4), numba.float64)

    result = cuda.local.array((4), numba.float64)
    G_i = cuda.local.array((4), numba.float64)
    G_k = cuda.local.array((4), numba.float64)



    zeros(sum_delx_delf, sum_delx_delf)
    zeros(sum_dely_delf, sum_dely_delf)

    x_i = globaldata[idx]['x']
    y_i = globaldata[idx]['y']

    nx = globaldata[idx]['nx']
    ny = globaldata[idx]['ny']

    tx = ny
    ty = -nx

    for itm in globaldata[idx]['xneg_conn'][:globaldata[idx]['xneg_nbhs']]:

        x_k = globaldata[itm]['x']
        y_k = globaldata[itm]['y']

        delx = x_k - x_i
        dely = y_k - y_i

        dels = delx*tx + dely*ty
        deln = delx*nx + dely*ny

        dist = (dels*dels + deln*deln) ** 0.5
        weights = dist**power

        dels_weights = dels*weights
        deln_weights = deln*weights

        sum_delx_sqr = sum_delx_sqr + dels*dels_weights
        sum_dely_sqr = sum_dely_sqr + deln*deln_weights

        sum_delx_dely = sum_delx_dely + dels*deln_weights

        zeros(qtilde_i, qtilde_i)
        zeros(qtilde_k, qtilde_k)

        zeros(temp1, temp1)
        zeros(temp2, temp2)

        multiply(delx, globaldata[idx]['dq'][0], temp1)
        multiply(dely, globaldata[idx]['dq'][1], temp2)

        add(temp1, temp2, temp1)
        multiply(0.5, temp1, temp1)
        subtract(globaldata[idx]['q'], temp1, qtilde_i)

        zeros(temp1, temp1)
        zeros(temp2, temp2)

        multiply(delx, globaldata[itm]['dq'][0], temp1)
        multiply(dely, globaldata[itm]['dq'][1], temp2)

        add(temp1, temp2, temp1)
        multiply(0.5, temp1, temp1)
        subtract(globaldata[itm]['q'], temp1, qtilde_k)

        zeros(phi_i, phi_i)
        zeros(phi_k, phi_k)

        limiters_cuda.venkat_limiter(qtilde_i, globaldata, idx, vl_const, phi_i)

        limiters_cuda.venkat_limiter(qtilde_k, globaldata, itm, vl_const, phi_k)

        zeros(temp1, temp1)
        zeros(temp2, temp2)

        multiply(delx, globaldata[idx]['dq'][0], temp1)
        multiply(dely, globaldata[idx]['dq'][1], temp2)

        multiply(0.5, phi_i, phi_i)

        add(temp1, temp2, temp1)
        multiply_element_wise(temp1, phi_i, temp1)

        subtract(globaldata[idx]['q'], temp1, qtilde_i)

        zeros(temp1, temp1)
        zeros(temp2, temp2)

        multiply(delx, globaldata[itm]['dq'][0], temp1)
        multiply(dely, globaldata[itm]['dq'][1], temp2)

        multiply(0.5, phi_k, phi_k)

        add(temp1, temp2, temp1)
        multiply_element_wise(temp1, phi_k, temp1)

        subtract(globaldata[itm]['q'], temp1, qtilde_k)

        zeros(result, result)

        qtilde_to_primitive_cuda(qtilde_i, gamma, result)

        quadrant_fluxes_cuda.flux_quad_GxIV(nx, ny, result[0], result[1], result[2], result[3], G_i)

        qtilde_to_primitive_cuda(qtilde_k, gamma, result)

        quadrant_fluxes_cuda.flux_quad_GxIV(nx, ny, result[0], result[1], result[2], result[3], G_k)

        zeros(temp1, temp1)
        subtract(G_k, G_i, temp1)
        multiply(dels_weights, temp1, temp1)
        add(sum_delx_delf, temp1, sum_delx_delf)

        zeros(temp2, temp2)
        subtract(G_k, G_i, temp2)
        multiply(deln_weights, temp2, temp2)
        add(sum_dely_delf, temp2, sum_dely_delf)

    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1 / det

    zeros(store, store)
    zeros(temp1, temp1)

    multiply(sum_dely_sqr, sum_delx_delf, sum_delx_delf)
    multiply(sum_delx_dely, sum_dely_delf, sum_dely_delf)

    subtract(sum_delx_delf, sum_dely_delf, temp1)

    multiply(one_by_det, temp1, store)


@cuda.jit(device=True)
def outer_dGy_pos(globaldata, idx, power, vl_const, gamma, store):
 
    sum_delx_sqr = 0
    sum_dely_sqr = 0
    sum_delx_dely = 0

    sum_delx_delf = cuda.local.array((4), numba.float64)
    sum_dely_delf = cuda.local.array((4), numba.float64)

    qtilde_i = cuda.local.array((4), numba.float64)
    qtilde_k = cuda.local.array((4), numba.float64)

    phi_i = cuda.local.array((4), numba.float64)
    phi_k = cuda.local.array((4), numba.float64)

    temp1 = cuda.local.array((4), numba.float64)
    temp2 = cuda.local.array((4), numba.float64)

    result = cuda.local.array((4), numba.float64)
    G_i = cuda.local.array((4), numba.float64)
    G_k = cuda.local.array((4), numba.float64)



    zeros(sum_delx_delf, sum_delx_delf)
    zeros(sum_dely_delf, sum_dely_delf)

    x_i = globaldata[idx]['x']
    y_i = globaldata[idx]['y']

    nx = globaldata[idx]['nx']
    ny = globaldata[idx]['ny']

    tx = ny
    ty = -nx

    for itm in globaldata[idx]['ypos_conn'][:globaldata[idx]['ypos_nbhs']]:

        x_k = globaldata[itm]['x']
        y_k = globaldata[itm]['y']

        delx = x_k - x_i
        dely = y_k - y_i

        dels = delx*tx + dely*ty
        deln = delx*nx + dely*ny

        dist = (dels*dels + deln*deln) ** 0.5
        weights = dist**power

        dels_weights = dels*weights
        deln_weights = deln*weights

        sum_delx_sqr = sum_delx_sqr + dels*dels_weights
        sum_dely_sqr = sum_dely_sqr + deln*deln_weights

        sum_delx_dely = sum_delx_dely + dels*deln_weights

        zeros(qtilde_i, qtilde_i)
        zeros(qtilde_k, qtilde_k)

        zeros(temp1, temp1)
        zeros(temp2, temp2)

        multiply(delx, globaldata[idx]['dq'][0], temp1)
        multiply(dely, globaldata[idx]['dq'][1], temp2)

        add(temp1, temp2, temp1)
        multiply(0.5, temp1, temp1)
        subtract(globaldata[idx]['q'], temp1, qtilde_i)

        zeros(temp1, temp1)
        zeros(temp2, temp2)

        multiply(delx, globaldata[itm]['dq'][0], temp1)
        multiply(dely, globaldata[itm]['dq'][1], temp2)

        add(temp1, temp2, temp1)
        multiply(0.5, temp1, temp1)
        subtract(globaldata[itm]['q'], temp1, qtilde_k)

        zeros(phi_i, phi_i)
        zeros(phi_k, phi_k)

        limiters_cuda.venkat_limiter(qtilde_i, globaldata, idx, vl_const, phi_i)

        limiters_cuda.venkat_limiter(qtilde_k, globaldata, itm, vl_const, phi_k)

        zeros(temp1, temp1)
        zeros(temp2, temp2)

        multiply(delx, globaldata[idx]['dq'][0], temp1)
        multiply(dely, globaldata[idx]['dq'][1], temp2)

        multiply(0.5, phi_i, phi_i)

        add(temp1, temp2, temp1)
        multiply_element_wise(temp1, phi_i, temp1)

        subtract(globaldata[idx]['q'], temp1, qtilde_i)

        zeros(temp1, temp1)
        zeros(temp2, temp2)

        multiply(delx, globaldata[itm]['dq'][0], temp1)
        multiply(dely, globaldata[itm]['dq'][1], temp2)

        multiply(0.5, phi_k, phi_k)

        add(temp1, temp2, temp1)
        multiply_element_wise(temp1, phi_k, temp1)

        subtract(globaldata[itm]['q'], temp1, qtilde_k)

        zeros(result, result)

        qtilde_to_primitive_cuda(qtilde_i, gamma, result)

        split_fluxes_cuda.flux_Gyp(nx, ny, result[0], result[1], result[2], result[3], G_i)

        qtilde_to_primitive_cuda(qtilde_k, gamma, result)

        split_fluxes_cuda.flux_Gyp(nx, ny, result[0], result[1], result[2], result[3], G_k)

        zeros(temp1, temp1)
        subtract(G_k, G_i, temp1)
        multiply(dels_weights, temp1, temp1)
        add(sum_delx_delf, temp1, sum_delx_delf)

        zeros(temp2, temp2)
        subtract(G_k, G_i, temp2)
        multiply(deln_weights, temp2, temp2)
        add(sum_dely_delf, temp2, sum_dely_delf)

    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1 / det

    zeros(store, store)
    zeros(temp1, temp1)

    multiply(sum_delx_dely, sum_delx_delf, sum_delx_delf)
    multiply(sum_delx_sqr, sum_dely_delf, sum_dely_delf)

    subtract(sum_dely_delf, sum_delx_delf, temp1)

    multiply(one_by_det, temp1, store)
