push!(LOAD_PATH, pwd());
println("=== Compiling. ===\n");
using adjoint_main_module;
println("=== Compilation done. ===\n");
println("=== Starting Adjoint Run. ===\n");
main()
println("=== Finished Adjoint Run. ===\n");

