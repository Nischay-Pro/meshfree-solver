using Distributed
using ClusterManagers
# addprocs()
addprocs(LocalAffinityManager(np = parse(Int, ARGS[1]), mode = BALANCED, affinities = Int[]))
@everywhere push!(LOAD_PATH, pwd());
println("=== Compiling. ===\n");
@everywhere using main_module
println("=== Compiled ===");
# println(length(workers()))
main()
check_leaks()