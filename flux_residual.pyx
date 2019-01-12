# cython: profile=True
# cython: binding=True
# cython: linetrace=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1
cimport wall_fluxes
cimport interior_fluxes
cimport outer_fluxes
import numpy as np
cimport numpy as np

cdef list cal_flux_residual(list globaldata, list wallindices, list outerindices, list interiorindices, dict configData):
	cdef long itm
	cdef np.ndarray[np.float64_t] Gxp, Gxn, Gyp, Gyn, GTemp
	for itm in wallindices:
		Gxp = wall_fluxes.wall_dGx_pos(globaldata, itm, configData)
		Gxn = wall_fluxes.wall_dGx_neg(globaldata, itm, configData)
		Gyn = wall_fluxes.wall_dGy_neg(globaldata, itm, configData)

		GTemp = Gxp + Gxn + Gyn
		GTemp = GTemp * 2

		globaldata[itm].set_flux_res(GTemp)
		
	for itm in outerindices:
		Gxp = outer_fluxes.outer_dGx_pos(globaldata, itm, configData)
		Gxn = outer_fluxes.outer_dGx_neg(globaldata, itm, configData)
		Gyp = outer_fluxes.outer_dGy_pos(globaldata, itm, configData)

		GTemp = Gxp + Gxn + Gyp
		GTemp = GTemp * 2

		globaldata[itm].set_flux_res(GTemp)

	for itm in interiorindices:
		Gxp = interior_fluxes.interior_dGx_pos(globaldata, itm, configData)
		Gxn = interior_fluxes.interior_dGx_neg(globaldata, itm, configData)
		Gyp = interior_fluxes.interior_dGy_pos(globaldata, itm, configData)
		Gyn = interior_fluxes.interior_dGy_neg(globaldata, itm, configData)

		GTemp = Gxp + Gxn + Gyp + Gyn
		GTemp = GTemp * 2

		globaldata[itm].set_flux_res(GTemp)

	return globaldata