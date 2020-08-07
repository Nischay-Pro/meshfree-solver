using Zygote
using StaticArrays
using StructArrays
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

# Define Parameters

numPoints = 9600

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

# Read into globaldata

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

include("objective_function.jl")
export calculateTheta, compute_cl_cd_cm

configData = getConfig()
max_iters = parse(Int, ARGS[2])
file_name = string(ARGS[1])
format = configData["format"]["type"]
numPoints = returnFileLength(file_name)

println(numPoints)
globaldata = Array{Point,1}(undef, numPoints)
res_old = zeros(Float64, 1)
main_store = zeros(Float64, 62)

defprimal = getInitialPrimitive(configData)

println("Start Read")
if format == "quadtree"
    readFileQuadtree(file_name::String, globaldata, defprimal, numPoints)
elseif format == "old"
    readFile(file_name::String, globaldata, defprimal, numPoints)
end

globaldata = StructArray(globaldata)

function func1(num)
	num = num + 2
    return num
end

function fpi_solver(var, iter, idx)
	#@.globaldata.x = globaldata.x+7
	#return globaldata.x[3]
	var = var^2
	# @timeit to "func1" begin
	# 	func1(var)
	# end
	return var
end

#check = fpi_solver(globaldata, 10, 3)
#println(check)
#print(globaldata)

iter = 10
idx = 3
var1 = 4
grad = gradient(fpi_solver, var1, iter, idx)
print(grad)