import operator
import wall_fluxes
import wall_fluxes_cuda
import interior_fluxes
import interior_fluxes_cuda
import outer_fluxes
import outer_fluxes_cuda
import config
import numpy as np
import numba
from numba import cuda
from cuda_func import add, zeros, multiply

def cal_flux_residual(globaldata, wallindices, outerindices, interiorindices, configData):
	for itm in wallindices:
		Gxp = wall_fluxes.wall_dGx_pos(globaldata, itm, configData)
		Gxn = wall_fluxes.wall_dGx_neg(globaldata, itm, configData)
		Gyn = wall_fluxes.wall_dGy_neg(globaldata, itm, configData)

		GTemp = np.array(Gxp) + np.array(Gxn) + np.array(Gyn)
		GTemp = GTemp * 2

		globaldata[itm].flux_res = GTemp
		
	for itm in outerindices:
		Gxp = outer_fluxes.outer_dGx_pos(globaldata, itm, configData)
		Gxn = outer_fluxes.outer_dGx_neg(globaldata, itm, configData)
		Gyp = outer_fluxes.outer_dGy_pos(globaldata, itm, configData)

		GTemp = np.array(Gxp) + np.array(Gxn) + np.array(Gyp)

		globaldata[itm].flux_res = GTemp

	for itm in interiorindices:
		Gxp = interior_fluxes.interior_dGx_pos(globaldata, itm, configData)
		Gxn = interior_fluxes.interior_dGx_neg(globaldata, itm, configData)
		Gyp = interior_fluxes.interior_dGy_pos(globaldata, itm, configData)
		Gyn = interior_fluxes.interior_dGy_neg(globaldata, itm, configData)

		GTemp = np.array(Gxp) + np.array(Gxn) + np.array(Gyp) + np.array(Gyn)

		globaldata[itm].flux_res = GTemp

	return globaldata

@cuda.jit()
def cal_flux_residual_cuda_kernel(globaldata, power, vl_const, gamma):
	tx = cuda.threadIdx.x
	bx = cuda.blockIdx.x
	bw = cuda.blockDim.x
	idx =  bx * bw + tx
	if idx > 0 and idx < len(globaldata):
		itm = globaldata[idx]
		flag_1 = itm['flag_1']
		if flag_1 == 1:

			Gxp = cuda.local.array((4), dtype=numba.float64)
			Gxn = cuda.local.array((4), dtype=numba.float64)
			Gyn = cuda.local.array((4), dtype=numba.float64)
			GTemp = cuda.local.array((4), dtype=numba.float64)

			zeros(Gxp, Gxp)
			zeros(Gxn, Gxn)
			zeros(Gyn, Gyn)
			zeros(GTemp, GTemp)

			x_i = globaldata[idx]['x']
			y_i = globaldata[idx]['y']

			wall_fluxes_cuda.wall_dGx_pos(globaldata, idx, power, vl_const, gamma, Gxp)
			wall_fluxes_cuda.wall_dGx_neg(globaldata, idx, power, vl_const, gamma, Gxn)
			wall_fluxes_cuda.wall_dGy_neg(globaldata, idx, power, vl_const, gamma, Gyn)

			add(Gxp, GTemp, GTemp)
			add(Gxn, GTemp, GTemp)
			add(Gyn, GTemp, GTemp)

			multiply(2, GTemp, GTemp)

			globaldata[itm]['flux_res'][0] = GTemp[0]
			globaldata[itm]['flux_res'][1] = GTemp[1]
			globaldata[itm]['flux_res'][2] = GTemp[2]
			globaldata[itm]['flux_res'][3] = GTemp[3]

		elif flag_1 == 2:

			Gxp = cuda.local.array((4), dtype=numba.float64)
			Gxn = cuda.local.array((4), dtype=numba.float64)
			Gyp = cuda.local.array((4), dtype=numba.float64)
			Gyn = cuda.local.array((4), dtype=numba.float64)
			GTemp = cuda.local.array((4), dtype=numba.float64)

			zeros(Gxp, Gxp)
			zeros(Gxn, Gxn)
			zeros(Gyn, Gyn)
			zeros(Gyp, Gyp)
			zeros(GTemp, GTemp)

			interior_fluxes_cuda.interior_dGx_pos(globaldata, idx, power, vl_const, gamma, Gxp)
			interior_fluxes_cuda.interior_dGx_neg(globaldata, idx, power, vl_const, gamma, Gxn)
			interior_fluxes_cuda.interior_dGy_neg(globaldata, idx, power, vl_const, gamma, Gyp)
			interior_fluxes_cuda.interior_dGy_neg(globaldata, idx, power, vl_const, gamma, Gyn)

			add(Gxp, GTemp, GTemp)
			add(Gxn, GTemp, GTemp)
			add(Gyp, GTemp, GTemp)
			add(Gyn, GTemp, GTemp)

			globaldata[itm]['flux_res'][0] = GTemp[0]
			globaldata[itm]['flux_res'][1] = GTemp[1]
			globaldata[itm]['flux_res'][2] = GTemp[2]
			globaldata[itm]['flux_res'][3] = GTemp[3]

		elif flag_1 == 3:

			Gxp = cuda.local.array((4), dtype=numba.float64)
			Gxn = cuda.local.array((4), dtype=numba.float64)
			Gyp = cuda.local.array((4), dtype=numba.float64)
			GTemp = cuda.local.array((4), dtype=numba.float64)

			zeros(Gxp, Gxp)
			zeros(Gxn, Gxn)
			zeros(Gyp, Gyp)
			zeros(GTemp, GTemp)

			outer_fluxes_cuda.outer_dGx_pos(globaldata, idx, power, vl_const, gamma, Gxp)
			outer_fluxes_cuda.outer_dGx_neg(globaldata, idx, power, vl_const, gamma, Gxn)
			outer_fluxes_cuda.outer_dGy_pos(globaldata, idx, power, vl_const, gamma, Gyn)

			add(Gxp, GTemp, GTemp)
			add(Gxn, GTemp, GTemp)
			add(Gyp, GTemp, GTemp)

			globaldata[itm]['flux_res'][0] = GTemp[0]
			globaldata[itm]['flux_res'][1] = GTemp[1]
			globaldata[itm]['flux_res'][2] = GTemp[2]
			globaldata[itm]['flux_res'][3] = GTemp[3]


@cuda.jit('float64[:](float64[:], int32, float64, float64, float64, float64[:])', device=True, inline=True)
def wall_dGx_pos(globaldata, idx, power, vl_const, gamma, store):

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

        dist = math.sqrt(dels*dels + deln*deln)
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
        limiters_cuda.venkat_limiter(qtilde_k, globaldata, idx, vl_const, phi_k)

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

        result = cuda.local.array((4), dtype=numba.float64)
        zeros(result, result)

        qtilde_to_primitive_cuda(qtilde_i, gamma, result)

        G_i = cuda.local.array((4), dtype=numba.float64)

        quadrant_fluxes_cuda.flux_quad_GxII(nx, ny, result[0], result[1], result[2], result[3], G_i)

        qtilde_to_primitive_cuda(qtilde_k, gamma, result)

        G_k = cuda.local.array((4), dtype=numba.float64)

        quadrant_fluxes_cuda.flux_quad_GxII(nx, ny, result[0], result[1], result[2], result[3], G_k)

        zeros(temp1, temp1)
        subtract(G_k, G_i, temp1)
        multiply(dels_weights, temp1, temp1)
        add(sum_delx_delf, temp1, sum_delx_delf)

        zeros(temp2, temp2)
        subtract(G_k, G_i, temp2)
        multiply(deln_weights, temp2, temp2)
        add(sum_dely_delf, temp1, sum_dely_delf)

    det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
    one_by_det = 1 / det

    zeros(store, store)
    zeros(temp1, temp1)

    multiply(sum_dely_sqr, sum_delx_delf, sum_delx_delf)
    multiply(sum_delx_dely, sum_dely_delf, sum_dely_delf)

    subtract(sum_delx_delf, sum_dely_delf, temp1)

    multiply(one_by_det, temp1, store)

    return store