import math
import limiters_cuda
import split_fluxes_cuda
import quadrant_fluxes_cuda
import numba
from numba import cuda
from cuda_func import add, zeros, multiply, qtilde_to_primitive_cuda, subtract, multiply_element_wise_shared
from operator import add as addop, sub as subop

@cuda.jit(device=True)
def outer_dGx_pos(x, y, nx_gpu, ny_gpu, min_dist, nbhs, conn, xpos_nbhs, xpos_conn, xneg_nbhs, xneg_conn, ypos_nbhs, ypos_conn, yneg_nbhs, yneg_conn, prim, q, maxminq, dq, flux_res, idx, power, vl_const, gamma, store, shared, sum_delx_delf, sum_dely_delf, qtilde_shared):

    sum_delx_sqr = 0
    sum_dely_sqr = 0
    sum_delx_dely = 0

    x_i = x[idx]
    y_i = y[idx]

    nx = nx_gpu[idx]
    ny = ny_gpu[idx]

    tx = ny
    ty = -nx

    for i in range(4):
        sum_delx_delf[cuda.threadIdx.x + cuda.blockDim.x * i] = 0
        sum_dely_delf[cuda.threadIdx.x + cuda.blockDim.x * i] = 0

    for itm in xpos_conn[idx][:xpos_nbhs[idx]]:

        x_k = x[itm]
        y_k = y[itm]

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

        limiters_cuda.venkat_limiter(qtilde_shared, q, maxminq, dq, nbhs, conn, x, y, min_dist, idx, vl_const, shared, delx, dely, gamma)
        quadrant_fluxes_cuda.flux_quad_GxIII(nx, ny, shared, addop)

        limiters_cuda.venkat_limiter(qtilde_shared, q, maxminq, dq, nbhs, conn, x, y, min_dist, itm, vl_const, shared, delx, dely, gamma)
        quadrant_fluxes_cuda.flux_quad_GxIII(nx, ny, shared, subop)

        for i in range(4):
            sum_delx_delf[cuda.threadIdx.x + cuda.blockDim.x * i] += dels_weights * shared[cuda.threadIdx.x + cuda.blockDim.x * i]
            sum_dely_delf[cuda.threadIdx.x + cuda.blockDim.x * i] += deln_weights * shared[cuda.threadIdx.x + cuda.blockDim.x * i]

    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1 / det

    for i in range(4):
        store[cuda.threadIdx.x + cuda.blockDim.x * i] += one_by_det * (sum_dely_sqr * sum_delx_delf[cuda.threadIdx.x + cuda.blockDim.x * i] - sum_delx_dely * sum_dely_delf[cuda.threadIdx.x + cuda.blockDim.x * i])

@cuda.jit(device=True)
def outer_dGx_neg(x, y, nx_gpu, ny_gpu, min_dist, nbhs, conn, xpos_nbhs, xpos_conn, xneg_nbhs, xneg_conn, ypos_nbhs, ypos_conn, yneg_nbhs, yneg_conn, prim, q, maxminq, dq, flux_res, idx, power, vl_const, gamma, store, shared, sum_delx_delf, sum_dely_delf, qtilde_shared):

    sum_delx_sqr = 0
    sum_dely_sqr = 0
    sum_delx_dely = 0

    x_i = x[idx]
    y_i = y[idx]

    nx = nx_gpu[idx]
    ny = ny_gpu[idx]

    tx = ny
    ty = -nx

    for i in range(4):
        sum_delx_delf[cuda.threadIdx.x + cuda.blockDim.x * i] = 0
        sum_dely_delf[cuda.threadIdx.x + cuda.blockDim.x * i] = 0

    for itm in xneg_conn[idx][:xneg_nbhs[idx]]:

        x_k = x[itm]
        y_k = y[itm]

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

        limiters_cuda.venkat_limiter(qtilde_shared, q, maxminq, dq, nbhs, conn, x, y, min_dist, idx, vl_const, shared, delx, dely, gamma)
        quadrant_fluxes_cuda.flux_quad_GxIV(nx, ny, shared, addop)

        limiters_cuda.venkat_limiter(qtilde_shared, q, maxminq, dq, nbhs, conn, x, y, min_dist, itm, vl_const, shared, delx, dely, gamma)
        quadrant_fluxes_cuda.flux_quad_GxIV(nx, ny, shared, subop)

        for i in range(4):
            sum_delx_delf[cuda.threadIdx.x + cuda.blockDim.x * i] += dels_weights * shared[cuda.threadIdx.x + cuda.blockDim.x * i]
            sum_dely_delf[cuda.threadIdx.x + cuda.blockDim.x * i] += deln_weights * shared[cuda.threadIdx.x + cuda.blockDim.x * i]

    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1 / det

    for i in range(4):
        store[cuda.threadIdx.x + cuda.blockDim.x * i] += one_by_det * (sum_dely_sqr * sum_delx_delf[cuda.threadIdx.x + cuda.blockDim.x * i] - sum_delx_dely * sum_dely_delf[cuda.threadIdx.x + cuda.blockDim.x * i])

@cuda.jit(device=True)
def outer_dGy_pos(x, y, nx_gpu, ny_gpu, min_dist, nbhs, conn, xpos_nbhs, xpos_conn, xneg_nbhs, xneg_conn, ypos_nbhs, ypos_conn, yneg_nbhs, yneg_conn, prim, q, maxminq, dq, flux_res, idx, power, vl_const, gamma, store, shared, sum_delx_delf, sum_dely_delf, qtilde_shared):
 
    sum_delx_sqr = 0
    sum_dely_sqr = 0
    sum_delx_dely = 0

    x_i = x[idx]
    y_i = y[idx]

    nx = nx_gpu[idx]
    ny = ny_gpu[idx]

    tx = ny
    ty = -nx

    for i in range(4):
        sum_delx_delf[cuda.threadIdx.x + cuda.blockDim.x * i] = 0
        sum_dely_delf[cuda.threadIdx.x + cuda.blockDim.x * i] = 0

    for itm in ypos_conn[idx][:ypos_nbhs[idx]]:

        x_k = x[itm]
        y_k = y[itm]

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

        limiters_cuda.venkat_limiter(qtilde_shared, q, maxminq, dq, nbhs, conn, x, y, min_dist, idx, vl_const, shared, delx, dely, gamma)
        split_fluxes_cuda.flux_Gyp(nx, ny, shared, addop)

        limiters_cuda.venkat_limiter(qtilde_shared, q, maxminq, dq, nbhs, conn, x, y, min_dist, itm, vl_const, shared, delx, dely, gamma)
        split_fluxes_cuda.flux_Gyp(nx, ny, shared, subop)

        for i in range(4):
            sum_delx_delf[cuda.threadIdx.x + cuda.blockDim.x * i] += dels_weights * shared[cuda.threadIdx.x + cuda.blockDim.x * i]
            sum_dely_delf[cuda.threadIdx.x + cuda.blockDim.x * i] += deln_weights * shared[cuda.threadIdx.x + cuda.blockDim.x * i]

    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1 / det

    for i in range(4):
        store[cuda.threadIdx.x + cuda.blockDim.x * i] += one_by_det * (sum_delx_sqr * sum_dely_delf[cuda.threadIdx.x + cuda.blockDim.x * i] - sum_delx_dely * sum_delx_delf[cuda.threadIdx.x + cuda.blockDim.x * i])