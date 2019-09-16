import math
import limiters_cuda
import split_fluxes_cuda
import quadrant_fluxes_cuda
import numba
from numba import cuda
from cuda_func import add, zeros, multiply, qtilde_to_primitive_cuda, subtract, multiply_element_wise_shared
from operator import add as addop, sub as subop

@cuda.jit(device=True, inline=True)
def outer_dGx_pos(globaldata, idx, power, vl_const, gamma, store, shared):

    sum_delx_sqr = 0
    sum_dely_sqr = 0
    sum_delx_dely = 0

    sum_delx_delf = cuda.local.array((4), numba.float64)
    sum_dely_delf = cuda.local.array((4), numba.float64)

    qtilde_i = cuda.local.array((4), numba.float64)
    qtilde_k = cuda.local.array((4), numba.float64)

    temp1 = cuda.local.array((4), numba.float64)
    temp2 = cuda.local.array((4), numba.float64)

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

        limiters_cuda.venkat_limiter(qtilde_i, globaldata, idx, vl_const, shared)

        zeros(temp1, temp1)
        zeros(temp2, temp2)

        multiply(delx, globaldata[idx]['dq'][0], temp1)
        multiply(dely, globaldata[idx]['dq'][1], temp2)

        add(temp1, temp2, temp1)
        multiply_element_wise_shared(temp1, shared, temp1)

        subtract(globaldata[idx]['q'], temp1, qtilde_i)

        zeros(temp1, temp1)
        zeros(temp2, temp2)

        limiters_cuda.venkat_limiter(qtilde_k, globaldata, itm, vl_const, shared)

        multiply(delx, globaldata[itm]['dq'][0], temp1)
        multiply(dely, globaldata[itm]['dq'][1], temp2)

        add(temp1, temp2, temp1)
        multiply_element_wise_shared(temp1, shared, temp1)

        subtract(globaldata[itm]['q'], temp1, qtilde_k)

        qtilde_to_primitive_cuda(qtilde_i, gamma, shared)

        shared[cuda.threadIdx.x], shared[cuda.threadIdx.x + cuda.blockDim.x], shared[cuda.threadIdx.x + cuda.blockDim.x * 2], shared[cuda.threadIdx.x + cuda.blockDim.x * 3] = 0, 0, 0, 0

        quadrant_fluxes_cuda.flux_quad_GxIII(nx, ny, shared, addop)

        qtilde_to_primitive_cuda(qtilde_k, gamma, shared)

        quadrant_fluxes_cuda.flux_quad_GxIII(nx, ny, shared, subop)

        temp1[0] = shared[cuda.threadIdx.x]
        temp1[1] = shared[cuda.threadIdx.x + cuda.blockDim.x]
        temp1[2] = shared[cuda.threadIdx.x + cuda.blockDim.x * 2]
        temp1[3] = shared[cuda.threadIdx.x + cuda.blockDim.x * 3]

        multiply(dels_weights, temp1, temp1)
        add(sum_delx_delf, temp1, sum_delx_delf)

        temp2[0] = shared[cuda.threadIdx.x]
        temp2[1] = shared[cuda.threadIdx.x + cuda.blockDim.x]
        temp2[2] = shared[cuda.threadIdx.x + cuda.blockDim.x * 2]
        temp2[3] = shared[cuda.threadIdx.x + cuda.blockDim.x * 3]

        multiply(deln_weights, temp2, temp2)
        add(sum_dely_delf, temp2, sum_dely_delf)

    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1 / det

    zeros(temp1, temp1)

    multiply(sum_dely_sqr, sum_delx_delf, sum_delx_delf)
    multiply(sum_delx_dely, sum_dely_delf, sum_dely_delf)

    subtract(sum_delx_delf, sum_dely_delf, temp1)

    store[cuda.threadIdx.x] += one_by_det * temp1[0]
    store[cuda.threadIdx.x + cuda.blockDim.x] += one_by_det * temp1[1]
    store[cuda.threadIdx.x + cuda.blockDim.x * 2] += one_by_det * temp1[2]
    store[cuda.threadIdx.x + cuda.blockDim.x * 3] += one_by_det * temp1[3]

@cuda.jit(device=True, inline=True)
def outer_dGx_neg(globaldata, idx, power, vl_const, gamma, store, shared):

    sum_delx_sqr = 0
    sum_dely_sqr = 0
    sum_delx_dely = 0

    sum_delx_delf = cuda.local.array((4), numba.float64)
    sum_dely_delf = cuda.local.array((4), numba.float64)

    qtilde_i = cuda.local.array((4), numba.float64)
    qtilde_k = cuda.local.array((4), numba.float64)

    temp1 = cuda.local.array((4), numba.float64)
    temp2 = cuda.local.array((4), numba.float64)

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

        limiters_cuda.venkat_limiter(qtilde_i, globaldata, idx, vl_const, shared)

        zeros(temp1, temp1)
        zeros(temp2, temp2)

        multiply(delx, globaldata[idx]['dq'][0], temp1)
        multiply(dely, globaldata[idx]['dq'][1], temp2)

        add(temp1, temp2, temp1)
        multiply_element_wise_shared(temp1, shared, temp1)

        subtract(globaldata[idx]['q'], temp1, qtilde_i)

        zeros(temp1, temp1)
        zeros(temp2, temp2)

        limiters_cuda.venkat_limiter(qtilde_k, globaldata, itm, vl_const, shared)

        multiply(delx, globaldata[itm]['dq'][0], temp1)
        multiply(dely, globaldata[itm]['dq'][1], temp2)

        add(temp1, temp2, temp1)
        multiply_element_wise_shared(temp1, shared, temp1)

        subtract(globaldata[itm]['q'], temp1, qtilde_k)

        qtilde_to_primitive_cuda(qtilde_i, gamma, shared)

        shared[cuda.threadIdx.x], shared[cuda.threadIdx.x + cuda.blockDim.x], shared[cuda.threadIdx.x + cuda.blockDim.x * 2], shared[cuda.threadIdx.x + cuda.blockDim.x * 3] = 0, 0, 0, 0

        quadrant_fluxes_cuda.flux_quad_GxIV(nx, ny, shared, addop)

        qtilde_to_primitive_cuda(qtilde_k, gamma, shared)

        quadrant_fluxes_cuda.flux_quad_GxIV(nx, ny, shared, subop)

        temp1[0] = shared[cuda.threadIdx.x]
        temp1[1] = shared[cuda.threadIdx.x + cuda.blockDim.x]
        temp1[2] = shared[cuda.threadIdx.x + cuda.blockDim.x * 2]
        temp1[3] = shared[cuda.threadIdx.x + cuda.blockDim.x * 3]

        multiply(dels_weights, temp1, temp1)
        add(sum_delx_delf, temp1, sum_delx_delf)

        temp2[0] = shared[cuda.threadIdx.x]
        temp2[1] = shared[cuda.threadIdx.x + cuda.blockDim.x]
        temp2[2] = shared[cuda.threadIdx.x + cuda.blockDim.x * 2]
        temp2[3] = shared[cuda.threadIdx.x + cuda.blockDim.x * 3]

        multiply(deln_weights, temp2, temp2)
        add(sum_dely_delf, temp2, sum_dely_delf)

    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1 / det

    zeros(temp1, temp1)

    multiply(sum_dely_sqr, sum_delx_delf, sum_delx_delf)
    multiply(sum_delx_dely, sum_dely_delf, sum_dely_delf)

    subtract(sum_delx_delf, sum_dely_delf, temp1)

    store[cuda.threadIdx.x] += one_by_det * temp1[0]
    store[cuda.threadIdx.x + cuda.blockDim.x] += one_by_det * temp1[1]
    store[cuda.threadIdx.x + cuda.blockDim.x * 2] += one_by_det * temp1[2]
    store[cuda.threadIdx.x + cuda.blockDim.x * 3] += one_by_det * temp1[3]

@cuda.jit(device=True, inline=True)
def outer_dGy_pos(globaldata, idx, power, vl_const, gamma, store, shared):
 
    sum_delx_sqr = 0
    sum_dely_sqr = 0
    sum_delx_dely = 0

    sum_delx_delf = cuda.local.array((4), numba.float64)
    sum_dely_delf = cuda.local.array((4), numba.float64)

    qtilde_i = cuda.local.array((4), numba.float64)
    qtilde_k = cuda.local.array((4), numba.float64)

    temp1 = cuda.local.array((4), numba.float64)
    temp2 = cuda.local.array((4), numba.float64)
    
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

        limiters_cuda.venkat_limiter(qtilde_i, globaldata, idx, vl_const, shared)

        zeros(temp1, temp1)
        zeros(temp2, temp2)

        multiply(delx, globaldata[idx]['dq'][0], temp1)
        multiply(dely, globaldata[idx]['dq'][1], temp2)

        add(temp1, temp2, temp1)
        multiply_element_wise_shared(temp1, shared, temp1)

        subtract(globaldata[idx]['q'], temp1, qtilde_i)

        zeros(temp1, temp1)
        zeros(temp2, temp2)

        limiters_cuda.venkat_limiter(qtilde_k, globaldata, itm, vl_const, shared)

        multiply(delx, globaldata[itm]['dq'][0], temp1)
        multiply(dely, globaldata[itm]['dq'][1], temp2)

        add(temp1, temp2, temp1)
        multiply_element_wise_shared(temp1, shared, temp1)

        subtract(globaldata[itm]['q'], temp1, qtilde_k)

        qtilde_to_primitive_cuda(qtilde_i, gamma, shared)

        shared[cuda.threadIdx.x], shared[cuda.threadIdx.x + cuda.blockDim.x], shared[cuda.threadIdx.x + cuda.blockDim.x * 2], shared[cuda.threadIdx.x + cuda.blockDim.x * 3] = 0, 0, 0, 0

        split_fluxes_cuda.flux_Gyp(nx, ny, shared, addop)

        qtilde_to_primitive_cuda(qtilde_k, gamma, shared)

        split_fluxes_cuda.flux_Gyp(nx, ny, shared, subop)

        temp1[0] = shared[cuda.threadIdx.x]
        temp1[1] = shared[cuda.threadIdx.x + cuda.blockDim.x]
        temp1[2] = shared[cuda.threadIdx.x + cuda.blockDim.x * 2]
        temp1[3] = shared[cuda.threadIdx.x + cuda.blockDim.x * 3]

        multiply(dels_weights, temp1, temp1)
        add(sum_delx_delf, temp1, sum_delx_delf)

        temp2[0] = shared[cuda.threadIdx.x]
        temp2[1] = shared[cuda.threadIdx.x + cuda.blockDim.x]
        temp2[2] = shared[cuda.threadIdx.x + cuda.blockDim.x * 2]
        temp2[3] = shared[cuda.threadIdx.x + cuda.blockDim.x * 3]

        multiply(deln_weights, temp2, temp2)
        add(sum_dely_delf, temp2, sum_dely_delf)

    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1 / det

    zeros(temp1, temp1)

    multiply(sum_delx_dely, sum_delx_delf, sum_delx_delf)
    multiply(sum_delx_sqr, sum_dely_delf, sum_dely_delf)

    subtract(sum_dely_delf, sum_delx_delf, temp1)

    store[cuda.threadIdx.x] += one_by_det * temp1[0]
    store[cuda.threadIdx.x + cuda.blockDim.x] += one_by_det * temp1[1]
    store[cuda.threadIdx.x + cuda.blockDim.x * 2] += one_by_det * temp1[2]
    store[cuda.threadIdx.x + cuda.blockDim.x * 3] += one_by_det * temp1[3]