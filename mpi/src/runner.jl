push!(LOAD_PATH, pwd());
println("=== Compiling. ===\n");
using main_module;
println("=== Compilation done. ===\n");
main()