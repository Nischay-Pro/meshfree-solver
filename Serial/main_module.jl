__precompile__()

module main_module

using Profile
using ProgressMeter
using Printf
using TimerOutputs
# using DelimitedFiles
using Traceur

const to = TimerOutput()

mutable struct Point
    localID::Int64
    x::Float64
    y::Float64
    left::Int64
    right::Int64
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
    prim_old::Array{Float64,1}
end

# using PyCall

# const math = PyNULL()

# function __init__()
#     copy!(math, pyimport("math"))
# end

include("config.jl")
export getConfig

include("core.jl")
export getInitialPrimitive, getInitialPrimitive2, calculateNormals, calculateConnectivity, fpi_solver, q_var_derivatives

include("file.jl")
export returnFileLength, readFile

include("flux_residual.jl")
export cal_flux_residual, wallindices_flux_residual, outerindices_flux_residual, interiorindices_flux_residual

include("interior_fluxes.jl")
export interior_dGx_pos, interior_dGx_neg, interior_dGy_pos, interior_dGy_neg

include("limiters.jl")
export venkat_limiter, maximum, minimum, smallest_dist, min_q_values, qtilde_to_primitive

include("meshfree_solver.jl")
export main

include("objective_function.jl")
export calculateTheta, compute_cl_cd_cm

include("outer_fluxes.jl")
export outer_dGx_pos, outer_dGx_neg, outer_dGy_pos

include("point.jl")
export getxy, setConnectivity

include("quadrant_fluxes.jl")
export flux_quad_GxI, flux_quad_GxII, flux_quad_GxIII, flux_quad_GxIV

include("split_fluxes.jl")
export flux_Gxp, flux_Gxn, flux_Gyp, flux_Gyn, flux_Gx, flux_Gy

include("state_update.jl")
export func_delta, state_update, primitive_to_conserved, conserved_vector_Ubar

include("wall_fluxes.jl")
export wall_dGx_pos, wall_dGx_neg, wall_dGy_neg

end
