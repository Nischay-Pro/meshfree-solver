import wall_fluxes_mpi
import interior_fluxes_mpi
import outer_fluxes_mpi
import config
import numpy as np

def cal_flux_residual_mpi(globaldata_local, globaldata_ghost, wallindices, outerindices, interiorindices, configData):
	for itm in wallindices:
		Gxp = wall_fluxes_mpi.wall_dGx_pos(globaldata_local, globaldata_ghost, itm, configData)
		Gxn = wall_fluxes_mpi.wall_dGx_neg(globaldata_local, globaldata_ghost, itm, configData)
		Gyn = wall_fluxes_mpi.wall_dGy_neg(globaldata_local, globaldata_ghost, itm, configData)

		GTemp = Gxp + Gxn + Gyn
		GTemp = GTemp * 2
		
		globaldata_local[itm].flux_res = GTemp
		
	for itm in outerindices:

		Gxp = outer_fluxes_mpi.outer_dGx_pos(globaldata_local, globaldata_ghost, itm, configData)
		Gxn = outer_fluxes_mpi.outer_dGx_neg(globaldata_local, globaldata_ghost, itm, configData)
		Gyp = outer_fluxes_mpi.outer_dGy_pos(globaldata_local, globaldata_ghost, itm, configData)

		GTemp = Gxp + Gxn + Gyp

		globaldata_local[itm].flux_res = GTemp

	for itm in interiorindices:
		Gxp = interior_fluxes_mpi.interior_dGx_pos(globaldata_local, globaldata_ghost, itm, configData)
		Gxn = interior_fluxes_mpi.interior_dGx_neg(globaldata_local, globaldata_ghost, itm, configData)
		Gyp = interior_fluxes_mpi.interior_dGy_pos(globaldata_local, globaldata_ghost, itm, configData)
		Gyn = interior_fluxes_mpi.interior_dGy_neg(globaldata_local, globaldata_ghost, itm, configData)

		GTemp = Gxp + Gxn + Gyp + Gyn

		globaldata_local[itm].flux_res = GTemp

	return globaldata_local