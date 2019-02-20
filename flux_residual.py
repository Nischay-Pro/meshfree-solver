import wall_fluxes
import interior_fluxes
import outer_fluxes
import config
import numpy as np

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