using Distributed
using ClusterManagers
# addprocs()
#addprocs(LocalAffinityManager(np = parse(Int, ARGS[1]), mode = BALANCED, affinities = Int[]))
addprocs(SlurmManager(parse(Int, ARGS[1])), lazy=false, enable_threaded_blas=false)
@everywhere push!(LOAD_PATH, pwd());
println("=== Compiling. ===\n");
@everywhere using main_module
println("=== Compiled ===");
# println(length(workers()))
main()
for i in workers()
	rmprocs(i)
end
check_leaks()
