import math
import limiters_cuda
import split_fluxes_cuda
import quadrant_fluxes_cuda
import numba
from numba import cuda
from cuda_func import add, zeros, multiply, qtilde_to_primitive_cuda, subtract, multiply_element_wise_shared
from operator import add as addop, sub as subop

@cuda.jit(inline=True)
def wall_dGx_pos(x, y, nx_gpu, ny_gpu, flag_1_gpu, min_dist, nbhs, conn, xpos_nbhs, xpos_conn, prim, q, maxminq, dq, flux_res, power, vl_const, gamma, wall):

    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x
    idx =  bx * bw + tx

    if flag_1_gpu[idx] != wall:
        return
    
    if idx == 0 or idx >= len(x):
        return

    other_shared = cuda.shared.array(shape = (1024 * 2), dtype=numba.float64)
    sum_delx_delf = cuda.shared.array(shape = (256 * 2), dtype=numba.float64)
    sum_dely_delf = cuda.shared.array(shape = (256 * 2), dtype=numba.float64)
    qtilde_shared = cuda.shared.array(shape = (256 * 2), dtype=numba.float64)

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

        itm = conn[idx][itm]
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

        other_shared[cuda.threadIdx.x], other_shared[cuda.threadIdx.x + cuda.blockDim.x], other_shared[cuda.threadIdx.x + cuda.blockDim.x * 2], other_shared[cuda.threadIdx.x + cuda.blockDim.x * 3] = 0, 0, 0, 0

        limiters_cuda.venkat_limiter(qtilde_shared, q, maxminq, dq, nbhs, conn, x, y, min_dist, idx, vl_const, other_shared, delx, dely, gamma)
        quadrant_fluxes_cuda.flux_quad_GxII(nx, ny, other_shared, addop)

        limiters_cuda.venkat_limiter(qtilde_shared, q, maxminq, dq, nbhs, conn, x, y, min_dist, itm, vl_const, other_shared, delx, dely, gamma)
        quadrant_fluxes_cuda.flux_quad_GxII(nx, ny, other_shared, subop)

        for i in range(4):
            sum_delx_delf[cuda.threadIdx.x + cuda.blockDim.x * i] += dels_weights * other_shared[cuda.threadIdx.x + cuda.blockDim.x * i]
            sum_dely_delf[cuda.threadIdx.x + cuda.blockDim.x * i] += deln_weights * other_shared[cuda.threadIdx.x + cuda.blockDim.x * i]

    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1 / det

    for i in range(4):
        sum_delx_delf[cuda.threadIdx.x + cuda.blockDim.x * i] *= sum_dely_sqr
        sum_dely_delf[cuda.threadIdx.x + cuda.blockDim.x * i] *= sum_delx_dely

    for i in range(4):
        flux_res[idx][i] = 2 * one_by_det * (sum_delx_delf[cuda.threadIdx.x + cuda.blockDim.x * i] - sum_dely_delf[cuda.threadIdx.x + cuda.blockDim.x * i])

@cuda.jit(inline=True)
def wall_dGx_neg(x, y, nx_gpu, ny_gpu, flag_1_gpu, min_dist, nbhs, conn, xneg_nbhs, xneg_conn, prim, q, maxminq, dq, flux_res, power, vl_const, gamma, wall):

    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x
    idx =  bx * bw + tx

    if flag_1_gpu[idx] != wall:
        return
    
    if idx == 0 or idx >= len(x):
        return

    other_shared = cuda.shared.array(shape = (1024 * 2), dtype=numba.float64)
    sum_delx_delf = cuda.shared.array(shape = (256 * 2), dtype=numba.float64)
    sum_dely_delf = cuda.shared.array(shape = (256 * 2), dtype=numba.float64)
    qtilde_shared = cuda.shared.array(shape = (256 * 2), dtype=numba.float64)

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

        itm = conn[idx][itm]
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

        other_shared[cuda.threadIdx.x], other_shared[cuda.threadIdx.x + cuda.blockDim.x], other_shared[cuda.threadIdx.x + cuda.blockDim.x * 2], other_shared[cuda.threadIdx.x + cuda.blockDim.x * 3] = 0, 0, 0, 0

        limiters_cuda.venkat_limiter(qtilde_shared, q, maxminq, dq, nbhs, conn, x, y, min_dist, idx, vl_const, other_shared, delx, dely, gamma)
        quadrant_fluxes_cuda.flux_quad_GxI(nx, ny, other_shared, addop)

        limiters_cuda.venkat_limiter(qtilde_shared, q, maxminq, dq, nbhs, conn, x, y, min_dist, itm, vl_const, other_shared, delx, dely, gamma)
        quadrant_fluxes_cuda.flux_quad_GxI(nx, ny, other_shared, subop)

        for i in range(4):
            sum_delx_delf[cuda.threadIdx.x + cuda.blockDim.x * i] += dels_weights * other_shared[cuda.threadIdx.x + cuda.blockDim.x * i]
            sum_dely_delf[cuda.threadIdx.x + cuda.blockDim.x * i] += deln_weights * other_shared[cuda.threadIdx.x + cuda.blockDim.x * i]

    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1 / det

    for i in range(4):
        sum_delx_delf[cuda.threadIdx.x + cuda.blockDim.x * i] *= sum_dely_sqr
        sum_dely_delf[cuda.threadIdx.x + cuda.blockDim.x * i] *= sum_delx_dely

    for i in range(4):
        flux_res[idx][i] += 2 * one_by_det * (sum_delx_delf[cuda.threadIdx.x + cuda.blockDim.x * i] - sum_dely_delf[cuda.threadIdx.x + cuda.blockDim.x * i])

@cuda.jit(inline=True)
def wall_dGy_neg(x, y, nx_gpu, ny_gpu, flag_1_gpu, min_dist, nbhs, conn, yneg_nbhs, yneg_conn, prim, q, maxminq, dq, flux_res, power, vl_const, gamma, wall):

    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x
    idx =  bx * bw + tx

    if flag_1_gpu[idx] != wall:
        return
    
    if idx == 0 or idx >= len(x):
        return

    other_shared = cuda.shared.array(shape = (1024 * 2), dtype=numba.float64)
    sum_delx_delf = cuda.shared.array(shape = (256 * 2), dtype=numba.float64)
    sum_dely_delf = cuda.shared.array(shape = (256 * 2), dtype=numba.float64)
    qtilde_shared = cuda.shared.array(shape = (256 * 2), dtype=numba.float64)

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

    for itm in yneg_conn[idx][:yneg_nbhs[idx]]:

        itm = conn[idx][itm]
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

        other_shared[cuda.threadIdx.x], other_shared[cuda.threadIdx.x + cuda.blockDim.x], other_shared[cuda.threadIdx.x + cuda.blockDim.x * 2], other_shared[cuda.threadIdx.x + cuda.blockDim.x * 3] = 0, 0, 0, 0

        limiters_cuda.venkat_limiter(qtilde_shared, q, maxminq, dq, nbhs, conn, x, y, min_dist, idx, vl_const, other_shared, delx, dely, gamma)
        split_fluxes_cuda.flux_Gyn(nx, ny, other_shared, addop)

        limiters_cuda.venkat_limiter(qtilde_shared, q, maxminq, dq, nbhs, conn, x, y, min_dist, itm, vl_const, other_shared, delx, dely, gamma)
        split_fluxes_cuda.flux_Gyn(nx, ny, other_shared, subop)

        for i in range(4):
            sum_delx_delf[cuda.threadIdx.x + cuda.blockDim.x * i] += dels_weights * other_shared[cuda.threadIdx.x + cuda.blockDim.x * i]
            sum_dely_delf[cuda.threadIdx.x + cuda.blockDim.x * i] += deln_weights * other_shared[cuda.threadIdx.x + cuda.blockDim.x * i]

    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1 / det

    for i in range(4):
        sum_delx_delf[cuda.threadIdx.x + cuda.blockDim.x * i] *= sum_delx_dely
        sum_dely_delf[cuda.threadIdx.x + cuda.blockDim.x * i] *= sum_delx_sqr

    for i in range(4):
        flux_res[idx][i] += 2 * one_by_det * (sum_dely_delf[cuda.threadIdx.x + cuda.blockDim.x * i] - sum_delx_delf[cuda.threadIdx.x + cuda.blockDim.x * i])