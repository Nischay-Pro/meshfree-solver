__precompile__()

module main_module

using JSON
using Printf
using TimerOutputs
using DelimitedFiles
using CuArrays
using CUDAnative
using CUDAdrv

const to = TimerOutput()
# using PyCall

# const math = PyNULL()

# function __init__()
#     copy!(math, pyimport("math"))
# end

include("config.jl")
export getConfig

include("core_cuda.jl")
export getInitialPrimitive, getInitialPrimitive2, calculateNormals, calculateConnectivity,
fpi_solver_cuda, q_var_cuda_kernel, q_var_derivatives_kernel

# include("cuda_funcs.jl")
# export reduce_warp, reduce_block, reduce_grid, gpu_reduce

include("flux_residual_cuda.jl")
export cal_flux_residual_kernel

include("interior_fluxes_cuda.jl")
export interior_dGx_pos_kernel, interior_dGx_neg_kernel, interior_dGy_pos_kernel, interior_dGy_neg_kernel

include("limiters_cuda.jl")
export venkat_limiter_kernel_i, venkat_limiter_kernel_k, qtilde_to_primitive_kernel

include("meshfree_solver.jl")
export main

include("objective_function.jl")
export calculateTheta, compute_cl_cd_cm

include("objective_function_cuda.jl")
export compute_cl_cd_cm_kernel, calculateTheta

include("outer_fluxes_cuda.jl")
export outer_dGx_pos_kernel, outer_dGx_neg_kernel, outer_dGy_pos_kernel

include("point.jl")
export Point, setNormals, getxy, setConnectivity, convertToArray

include("quadrant_fluxes_cuda.jl")
export flux_quad_GxI_kernel, flux_quad_GxII_kernel, flux_quad_GxIII_kernel, flux_quad_GxIV_kernel

include("split_fluxes_cuda.jl")
export flux_Gxp_kernel, flux_Gxn_kernel, flux_Gyp_kernel, flux_Gyn_kernel, flux_Gx_kernel, flux_Gy_kernel

include("state_update_cuda.jl")
export func_delta_kernel, state_update_kernel

include("wall_fluxes_cuda.jl")
export wall_dGx_pos_kernel, wall_dGx_neg_kernel, wall_dGy_neg_kernel

end
