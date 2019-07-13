push!(LOAD_PATH, "/home/kumar/Julia/meshfree-solver_main/JULIA_MESHFREE_CUDA");
println("\n===\n");
using main_module;
println("\n===\n");
# for thread_types in ["32"]
    # ARGS[2] = thread_types
    main()
# end