import math
import limiters_cuda
import split_fluxes_cuda
import quadrant_fluxes_cuda
import numba
from numba import cuda
from cuda_func import add, zeros, multiply, qtilde_to_primitive_cuda, subtract, multiply_element_wise_shared
from operator import add as addop, sub as subop


@cuda.jit(device=True)
def wall_dGx_pos(globaldata, idx, power, vl_const, gamma, store, shared, sum_delx_delf, sum_dely_delf, qtilde_shared):

    sum_delx_sqr = 0
    sum_dely_sqr = 0
    sum_delx_dely = 0

    x_i = globaldata[idx]['x']
    y_i = globaldata[idx]['y']

    nx = globaldata[idx]['nx']
    ny = globaldata[idx]['ny']

    tx = ny
    ty = -nx

    for i in range(4):
        sum_delx_delf[cuda.threadIdx.x + cuda.blockDim.x * i] = 0
        sum_dely_delf[cuda.threadIdx.x + cuda.blockDim.x * i] = 0

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

        shared[cuda.threadIdx.x], shared[cuda.threadIdx.x + cuda.blockDim.x], shared[cuda.threadIdx.x + cuda.blockDim.x * 2], shared[cuda.threadIdx.x + cuda.blockDim.x * 3] = 0, 0, 0, 0

        limiters_cuda.venkat_limiter(qtilde_shared, globaldata, idx, vl_const, shared, delx, dely, gamma)
        quadrant_fluxes_cuda.flux_quad_GxII(nx, ny, shared, addop)

        limiters_cuda.venkat_limiter(qtilde_shared, globaldata, itm, vl_const, shared, delx, dely, gamma)
        quadrant_fluxes_cuda.flux_quad_GxII(nx, ny, shared, subop)

        for i in range(4):
            sum_delx_delf[cuda.threadIdx.x + cuda.blockDim.x * i] += dels_weights * shared[cuda.threadIdx.x + cuda.blockDim.x * i]
            sum_dely_delf[cuda.threadIdx.x + cuda.blockDim.x * i] += deln_weights * shared[cuda.threadIdx.x + cuda.blockDim.x * i]

    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1 / det

    for i in range(4):
        sum_delx_delf[cuda.threadIdx.x + cuda.blockDim.x * i] *= sum_dely_sqr
        sum_dely_delf[cuda.threadIdx.x + cuda.blockDim.x * i] *= sum_delx_dely

    for i in range(4):
        store[cuda.threadIdx.x + cuda.blockDim.x * i] += 2 * one_by_det * (sum_delx_delf[cuda.threadIdx.x + cuda.blockDim.x * i] - sum_dely_delf[cuda.threadIdx.x + cuda.blockDim.x * i])

@cuda.jit(device=True)
def wall_dGx_neg(globaldata, idx, power, vl_const, gamma, store, shared, sum_delx_delf, sum_dely_delf, qtilde_shared):

    sum_delx_sqr = 0
    sum_dely_sqr = 0
    sum_delx_dely = 0

    x_i = globaldata[idx]['x']
    y_i = globaldata[idx]['y']

    nx = globaldata[idx]['nx']
    ny = globaldata[idx]['ny']

    tx = ny
    ty = -nx

    for i in range(4):
        sum_delx_delf[cuda.threadIdx.x + cuda.blockDim.x * i] = 0
        sum_dely_delf[cuda.threadIdx.x + cuda.blockDim.x * i] = 0

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

        shared[cuda.threadIdx.x], shared[cuda.threadIdx.x + cuda.blockDim.x], shared[cuda.threadIdx.x + cuda.blockDim.x * 2], shared[cuda.threadIdx.x + cuda.blockDim.x * 3] = 0, 0, 0, 0

        limiters_cuda.venkat_limiter(qtilde_shared, globaldata, idx, vl_const, shared, delx, dely, gamma)
        quadrant_fluxes_cuda.flux_quad_GxI(nx, ny, shared, addop)

        limiters_cuda.venkat_limiter(qtilde_shared, globaldata, itm, vl_const, shared, delx, dely, gamma)
        quadrant_fluxes_cuda.flux_quad_GxI(nx, ny, shared, subop)

        for i in range(4):
            sum_delx_delf[cuda.threadIdx.x + cuda.blockDim.x * i] += dels_weights * shared[cuda.threadIdx.x + cuda.blockDim.x * i]
            sum_dely_delf[cuda.threadIdx.x + cuda.blockDim.x * i] += deln_weights * shared[cuda.threadIdx.x + cuda.blockDim.x * i]

    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1 / det

    for i in range(4):
        sum_delx_delf[cuda.threadIdx.x + cuda.blockDim.x * i] *= sum_dely_sqr
        sum_dely_delf[cuda.threadIdx.x + cuda.blockDim.x * i] *= sum_delx_dely

    for i in range(4):
        store[cuda.threadIdx.x + cuda.blockDim.x * i] += 2 * one_by_det * (sum_delx_delf[cuda.threadIdx.x + cuda.blockDim.x * i] - sum_dely_delf[cuda.threadIdx.x + cuda.blockDim.x * i])

@cuda.jit(device=True)
def wall_dGy_neg(globaldata, idx, power, vl_const, gamma, store, shared, sum_delx_delf, sum_dely_delf, qtilde_shared):

    sum_delx_sqr = 0
    sum_dely_sqr = 0
    sum_delx_dely = 0

    x_i = globaldata[idx]['x']
    y_i = globaldata[idx]['y']

    nx = globaldata[idx]['nx']
    ny = globaldata[idx]['ny']

    tx = ny
    ty = -nx

    for i in range(4):
        sum_delx_delf[cuda.threadIdx.x + cuda.blockDim.x * i] = 0
        sum_dely_delf[cuda.threadIdx.x + cuda.blockDim.x * i] = 0

    for itm in globaldata[idx]['yneg_conn'][:globaldata[idx]['yneg_nbhs']]:

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

        shared[cuda.threadIdx.x], shared[cuda.threadIdx.x + cuda.blockDim.x], shared[cuda.threadIdx.x + cuda.blockDim.x * 2], shared[cuda.threadIdx.x + cuda.blockDim.x * 3] = 0, 0, 0, 0

        limiters_cuda.venkat_limiter(qtilde_shared, globaldata, idx, vl_const, shared, delx, dely, gamma)
        split_fluxes_cuda.flux_Gyn(nx, ny, shared, addop)

        limiters_cuda.venkat_limiter(qtilde_shared, globaldata, itm, vl_const, shared, delx, dely, gamma)
        split_fluxes_cuda.flux_Gyn(nx, ny, shared, subop)

        for i in range(4):
            sum_delx_delf[cuda.threadIdx.x + cuda.blockDim.x * i] += dels_weights * shared[cuda.threadIdx.x + cuda.blockDim.x * i]
            sum_dely_delf[cuda.threadIdx.x + cuda.blockDim.x * i] += deln_weights * shared[cuda.threadIdx.x + cuda.blockDim.x * i]

    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1 / det

    for i in range(4):
        sum_delx_delf[cuda.threadIdx.x + cuda.blockDim.x * i] *= sum_delx_dely
        sum_dely_delf[cuda.threadIdx.x + cuda.blockDim.x * i] *= sum_delx_sqr

    for i in range(4):
        store[cuda.threadIdx.x + cuda.blockDim.x * i] += 2 * one_by_det * (sum_dely_delf[cuda.threadIdx.x + cuda.blockDim.x * i] - sum_delx_delf[cuda.threadIdx.x + cuda.blockDim.x * i])