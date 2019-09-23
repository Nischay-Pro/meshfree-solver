push!(LOAD_PATH, pwd());
println("=== Compiling ===\n");
using main_module;
println("=== Compilation done ===\n");
# for thread_types in ["8","16","32","64","128"]
    # ARGS[2] = thread_types
    main()
# end