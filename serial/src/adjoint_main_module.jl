__precompile__()

module adjoint_main_module

using BenchmarkTools
# using Cthulhu
# using CuArrays
# using CUDAnative
# using GPUifyLoops
# using LoopVectorization
# using Profile
using ProgressMeter
using Printf
using Setfield
using StaticArrays
using StructArrays
using TimerOutputs
using SpecialFunctions
# using DelimitedFiles
# using Traceur

# Import Zygote for the Adjoint AD 
using Zygote

const to = TimerOutput()

struct Point
    localID::Int32
    x::Float64
    y::Float64
    left::Int32
    right::Int32
    flag_1::Int8
    flag_2::Int8
    short_distance::Float64
    nbhs::Int8
    conn::SArray{Tuple{20},Int32,1,20}
    nx::Float64
    ny::Float64
    # Size 4 (Pressure, vx, vy, density) x numberpts
    prim::SArray{Tuple{4},Float64,1,4}
    flux_res::SArray{Tuple{4},Float64,1,4}
    # Size 4 (Pressure, vx, vy, density) x numberpts
    q::SArray{Tuple{4},Float64,1,4}
    # Size 2(x,y) 4(Pressure, vx, vy, density) numberpts
    dq1::SArray{Tuple{4},Float64,1,4}
    dq2::SArray{Tuple{4},Float64,1,4}
    entropy::Float64
    xpos_nbhs::Int8
    xneg_nbhs::Int8
    ypos_nbhs::Int8
    yneg_nbhs::Int8
    xpos_conn::SArray{Tuple{20},Int32,1,20}
    xneg_conn::SArray{Tuple{20},Int32,1,20}
    ypos_conn::SArray{Tuple{20},Int32,1,20}
    yneg_conn::SArray{Tuple{20},Int32,1,20}
    delta::Float64
    max_q::SArray{Tuple{4},Float64,1,4}
    min_q::SArray{Tuple{4},Float64,1,4}
    prim_old::SArray{Tuple{4},Float64,1,4}
end

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
export venkat_limiter, maximum, minimum, qtilde_to_primitive

include("adjoint_meshfree_solver.jl")
export main

include("objective_function.jl")
export calculateTheta, compute_cl_cd_cm

include("outer_fluxes.jl")
export outer_dGx_pos, outer_dGx_neg, outer_dGy_pos

include("point.jl")
export getxy

include("quadrant_fluxes.jl")
export flux_quad_GxI, flux_quad_GxII, flux_quad_GxIII, flux_quad_GxIV

include("split_fluxes.jl")
export flux_Gxp, flux_Gxn, flux_Gyp, flux_Gyn, flux_Gx, flux_Gy

include("state_update.jl")
export func_delta, state_update, primitive_to_conserved, conserved_vector_Ubar

include("wall_fluxes.jl")
export wall_dGx_pos, wall_dGx_neg, wall_dGy_neg

end
