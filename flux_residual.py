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

			wall_fluxes_cuda.wall_dGx_pos(globaldata, idx, power, vl_const, gamma, Gxp)
			wall_fluxes_cuda.wall_dGx_neg(globaldata, idx, power, vl_const, gamma, Gxn)
			wall_fluxes_cuda.wall_dGy_neg(globaldata, idx, power, vl_const, gamma, Gyn)

			add(Gxp, GTemp, GTemp)
			add(Gxn, GTemp, GTemp)
			add(Gyn, GTemp, GTemp)

			multiply(2, GTemp, GTemp)

			globaldata[idx]['flux_res'][0] = GTemp[0]
			globaldata[idx]['flux_res'][1] = GTemp[1]
			globaldata[idx]['flux_res'][2] = GTemp[2]
			globaldata[idx]['flux_res'][3] = GTemp[3]

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
			interior_fluxes_cuda.interior_dGy_pos(globaldata, idx, power, vl_const, gamma, Gyp)
			interior_fluxes_cuda.interior_dGy_neg(globaldata, idx, power, vl_const, gamma, Gyn)

			add(Gxp, GTemp, GTemp)
			add(Gxn, GTemp, GTemp)
			add(Gyp, GTemp, GTemp)
			add(Gyn, GTemp, GTemp)

			globaldata[idx]['flux_res'][0] = GTemp[0]
			globaldata[idx]['flux_res'][1] = GTemp[1]
			globaldata[idx]['flux_res'][2] = GTemp[2]
			globaldata[idx]['flux_res'][3] = GTemp[3]

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
			outer_fluxes_cuda.outer_dGy_pos(globaldata, idx, power, vl_const, gamma, Gyp)

			add(Gxp, GTemp, GTemp)
			add(Gxn, GTemp, GTemp)
			add(Gyp, GTemp, GTemp)

			globaldata[idx]['flux_res'][0] = GTemp[0]
			globaldata[idx]['flux_res'][1] = GTemp[1]
			globaldata[idx]['flux_res'][2] = GTemp[2]
			globaldata[idx]['flux_res'][3] = GTemp[3]