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

@cuda.jit(inline=True)
def cal_flux_residual_cuda_kernel(globaldata, power, vl_const, gamma, wall, interior, outer):
	tx = cuda.threadIdx.x
	bx = cuda.blockIdx.x
	bw = cuda.blockDim.x
	idx =  bx * bw + tx

	if idx > 0 and idx < len(globaldata):

		other_shared = cuda.shared.array(shape = (1024), dtype=numba.float64)
		flux_shared = cuda.shared.array(shape = (256), dtype=numba.float64)
		sum_delx_delf = cuda.shared.array(shape = (256), dtype=numba.float64)
		sum_dely_delf = cuda.shared.array(shape = (256), dtype=numba.float64)
		qtilde_shared = cuda.shared.array(shape = (256), dtype=numba.float64)

		zeros(other_shared, other_shared)
		zeros(flux_shared, flux_shared)
		
		itm = globaldata[idx]
		flag_1 = itm['flag_1']
		if flag_1 == wall:

			wall_fluxes_cuda.wall_dGx_pos(globaldata, idx, power, vl_const, gamma, flux_shared, other_shared, sum_delx_delf, sum_dely_delf, qtilde_shared)
			wall_fluxes_cuda.wall_dGx_neg(globaldata, idx, power, vl_const, gamma, flux_shared, other_shared, sum_delx_delf, sum_dely_delf, qtilde_shared)
			wall_fluxes_cuda.wall_dGy_neg(globaldata, idx, power, vl_const, gamma, flux_shared, other_shared, sum_delx_delf, sum_dely_delf, qtilde_shared)

			globaldata[idx]['flux_res'][0] = flux_shared[cuda.threadIdx.x]
			globaldata[idx]['flux_res'][1] = flux_shared[cuda.threadIdx.x + cuda.blockDim.x]
			globaldata[idx]['flux_res'][2] = flux_shared[cuda.threadIdx.x + cuda.blockDim.x * 2]
			globaldata[idx]['flux_res'][3] = flux_shared[cuda.threadIdx.x + cuda.blockDim.x * 3]

		elif flag_1 == interior:

			interior_fluxes_cuda.interior_dGx_pos(globaldata, idx, power, vl_const, gamma, flux_shared, other_shared, sum_delx_delf, sum_dely_delf, qtilde_shared)
			interior_fluxes_cuda.interior_dGx_neg(globaldata, idx, power, vl_const, gamma, flux_shared, other_shared, sum_delx_delf, sum_dely_delf, qtilde_shared)
			interior_fluxes_cuda.interior_dGy_pos(globaldata, idx, power, vl_const, gamma, flux_shared, other_shared, sum_delx_delf, sum_dely_delf, qtilde_shared)
			interior_fluxes_cuda.interior_dGy_neg(globaldata, idx, power, vl_const, gamma, flux_shared, other_shared, sum_delx_delf, sum_dely_delf, qtilde_shared)

			globaldata[idx]['flux_res'][0] = flux_shared[cuda.threadIdx.x]
			globaldata[idx]['flux_res'][1] = flux_shared[cuda.threadIdx.x + cuda.blockDim.x]
			globaldata[idx]['flux_res'][2] = flux_shared[cuda.threadIdx.x + cuda.blockDim.x * 2]
			globaldata[idx]['flux_res'][3] = flux_shared[cuda.threadIdx.x + cuda.blockDim.x * 3]

		elif flag_1 == outer:

			outer_fluxes_cuda.outer_dGx_pos(globaldata, idx, power, vl_const, gamma, flux_shared, other_shared, sum_delx_delf, sum_dely_delf, qtilde_shared)
			outer_fluxes_cuda.outer_dGx_neg(globaldata, idx, power, vl_const, gamma, flux_shared, other_shared, sum_delx_delf, sum_dely_delf, qtilde_shared)
			outer_fluxes_cuda.outer_dGy_pos(globaldata, idx, power, vl_const, gamma, flux_shared, other_shared, sum_delx_delf, sum_dely_delf, qtilde_shared)

			globaldata[idx]['flux_res'][0] = flux_shared[cuda.threadIdx.x]
			globaldata[idx]['flux_res'][1] = flux_shared[cuda.threadIdx.x + cuda.blockDim.x]
			globaldata[idx]['flux_res'][2] = flux_shared[cuda.threadIdx.x + cuda.blockDim.x * 2]
			globaldata[idx]['flux_res'][3] = flux_shared[cuda.threadIdx.x + cuda.blockDim.x * 3]