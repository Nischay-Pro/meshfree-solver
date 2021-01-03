__precompile__()

module main_module

# using CuArrays
# using CUDAdrv
# using CUDAnative
using CUDA
using DelimitedFiles
using JSON
using Printf
using ProgressMeter
using StaticArrays
using TimerOutputs

const to = TimerOutput()

mutable struct Point
    localID::Int32
    x::Float64
    y::Float64
    left::Int32
    right::Int32
    flag_1::Int8
    flag_2::Int8
    short_distance::Float64
    nbhs::Int8
    conn::Array{Int32,1}
    nx::Float64
    ny::Float64
    # Size 4 (Pressure, vx, vy, density) x numberpts
    prim::Array{Float64,1}
    flux_res::Array{Float64,1}
    # Size 4 (Pressure, vx, vy, density) x numberpts
    q::Array{Float64,1}
    # Size 2(x,y) 4(Pressure, vx, vy, density) numberpts
    dq::Array{Array{Float64,1},1}
    entropy::Float64
    xpos_nbhs::Int8
    xneg_nbhs::Int8
    ypos_nbhs::Int8
    yneg_nbhs::Int8
    xpos_conn::Array{Int32,1}
    xneg_conn::Array{Int32,1}
    ypos_conn::Array{Int32,1}
    yneg_conn::Array{Int32,1}
    delta::Float64
    max_q::Array{Float64,1}
    min_q::Array{Float64,1}
end


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

include("file.jl")
export returnFileLength, readFile

include("flux_residual_cuda.jl")
export cal_flux_residual_kernel

include("interior_fluxes_cuda.jl")
export interior_dGx_pos_kernel, interior_dGx_neg_kernel, interior_dGy_pos_kernel, interior_dGy_neg_kernel

include("limiters_cuda.jl")
export venkat_limiter_kernel_qtilde

include("meshfree_solver.jl")
export main

include("objective_function_cuda.jl")
export compute_cl_cd_cm, calculateTheta

include("outer_fluxes_cuda.jl")
export outer_dGx_pos_kernel, outer_dGx_neg_kernel, outer_dGy_pos_kernel

include("point.jl")
export Point, setNormals, getxy, setConnectivity

include("quadrant_fluxes_cuda.jl")
export flux_quad_GxI_kernel, flux_quad_GxII_kernel, flux_quad_GxIII_kernel, flux_quad_GxIV_kernel

include("split_fluxes_cuda.jl")
export flux_Gxp_kernel, flux_Gxn_kernel, flux_Gyp_kernel, flux_Gyn_kernel

include("state_update_cuda.jl")
export func_delta_kernel, state_update_kernel

include("wall_fluxes_cuda.jl")
export wall_dGx_pos_kernel, wall_dGx_neg_kernel, wall_dGy_neg_kernel

end
