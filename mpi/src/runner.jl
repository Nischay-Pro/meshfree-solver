push!(LOAD_PATH, pwd());
print("=== Compiling. ===\n");
using main_module;
print("=== Compilation done. ===\n");
main()