push!(LOAD_PATH, pwd());
println("=== Compiling. ===\n");
using main_module;
println("=== Compilation done. ===\n");
# for thread_types in ["32"]
    # ARGS[2] = thread_types
    main()
# end