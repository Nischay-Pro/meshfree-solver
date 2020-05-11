function main()
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)
    # print("Hello world, I am rank $(rank) of $(size)\n")
    # MPI.Barrier(comm)

    configData = getConfig()
    max_iters = parse(Int, ARGS[2])
    file_name = string(ARGS[1])
    file_name = specificCoreFile(file_name)
    format = configData["format"]["type"]
    numPoints = returnFileLength(file_name)
    localPoints, ghostPoints, partitionGhostLimit = 0, 0, 0

    if rank == 0
        println(numPoints, " for Rank:$rank")
    end
    globaldata = Array{Point,1}(undef, numPoints)
    res_old = zeros(Float64, 1)
    main_store = zeros(Float64, 62)
    defprimal = getInitialPrimitive(configData)

    #####
    ##### File Reading
    #####
    if rank == 0
        println("Start Read")
    end
    if format == "quadtree"
        readFileQuadtree(file_name::String, globaldata, defprimal)
    elseif format == "old"
        readFile(file_name::String, globaldata, defprimal)
    elseif format == "mpi"
        localPoints, ghostPoints, partitionGhostLimit = readFileMPIQuadtree(file_name::String, globaldata, defprimal, localPoints, ghostPoints)
    end

    foreignGhostPoints = zeros(Int, partitionGhostLimit, size)

    #####
    ##### Normals & Connectivity Generation
    #####
    interior::Int64 = configData["point"]["interior"]
    wall::Int64 = configData["point"]["wall"]
    outer::Int64 = configData["point"]["outer"]
    @showprogress 2 "Computing Normals" for idx in 1:localPoints
        placeNormals(globaldata, idx, configData, interior, wall, outer)
    end

    if rank == 0
        println("Start Connectivity Generation")
    end
    @showprogress 3 "Computing Connectivity" for idx in 1:localPoints
        calculateConnectivity(globaldata, idx)
    end

    set_local_to_ghost_ID(globaldata, foreignGhostPoints, localPoints, ghostPoints, rank, size, comm)
    # if rank == 0
    #     # for idx in 1:partitionGhostLimit
    #         print(foreignGhostPoints, "\n")
    #     # end
    # end

    #####
    ##### Config File Data
    #####
    main_store[53] = configData["core"]["power"]::Float64
    main_store[54] = configData["core"]["cfl"]::Float64 
    main_store[55] = configData["core"]["limiter_flag"]::Int64
    main_store[56] = configData["core"]["vl_const"]::Float64
    main_store[57] = configData["core"]["aoa"]::Float64
    main_store[58] = configData["core"]["mach"]::Float64
    main_store[59] = configData["core"]["gamma"]::Float64
    main_store[60] = configData["core"]["pr_inf"]::Float64
    main_store[61] = configData["core"]["rho_inf"]::Float64
    main_store[62] = calculateTheta(configData)::Float64

    globaldata = StructArray(globaldata)
    # println(typeof(globaldata))
    # println(typeof(globaldata.prim))
    # println(isbits(globaldata[1]))
    # println(isbits(globaldata.prim))
    # println(isbits(globaldata))
    # data = rand(Float32, 134217728)
    # @timeit to "transfer time" begin
    #     CuArray(data)
    #     gpu_data = CuArray(globaldata.prim)
    #     time = @belapsed unsafe_copyto!($(pointer(gpu_data)), $(pointer(globaldata.prim)), $(numPoints))
    #     println(time)
    #     println(Base.format_bytes(sizeof(globaldata.prim) / time) * "/s")
    # end
    # println(typeof(globaldata))
    # replace_storage(CuArray, globaldata)
    # println(typeof(globaldata))
    # replace_storage(Array, globaldata)
    # println(typeof(globaldata))
    
    #####
    ##### Core Running Code
    #####
    if rank == 0
        println(max_iters + 1)
        println(localPoints)
        println(ghostPoints)
    end
    function run_code(globaldata, configData, res_old, localPoints, ghostPoints, main_store, foreignGhostPoints, tempdq, requests_1, requests_2)
        for i in 1:max_iters
            fpi_solver(i, globaldata, configData, res_old, localPoints, ghostPoints, main_store, foreignGhostPoints, tempdq, requests_1, requests_2)
        end
    end

    function test_code(globaldata, configData, res_old, localPoints, ghostPoints, main_store, foreignGhostPoints)
        res_old[1] = 0.0
        tempdq = zeros(Float64, numPoints, 2, 4)
        requests_1 = Array{MPI.Request,1}(undef, ghostPoints)
        requests_2 = Array{MPI.Request,1}(undef, ghostPoints)
        @timeit to "nest 1" begin
            run_code(globaldata, configData, res_old, localPoints, ghostPoints, main_store, foreignGhostPoints, tempdq, requests_1, requests_2)
            MPI.Barrier(MPI.COMM_WORLD)
        end
    end

    test_code(globaldata, configData, res_old, localPoints, ghostPoints, main_store, foreignGhostPoints)
    localPointsArray = [localPoints]
    storePointsAll = zeros(Int, 1)
    MPI.Allreduce!(localPointsArray, storePointsAll, +, comm)
    # print(storePointsAll, "\n")

    #####
    ##### Output & Benchmark Files
    #####
    open("../results_mpi/timer_mpi" * string(storePointsAll[1]) * "_" * string(rank) * "_" * string(getConfig()["core"]["max_iters"]) *".txt", "w") do io
        print_timer(io, to)
    end

    # compute_cl_cd_cm(globaldata, configData, shapeptsidx)
    # println(globaldata[1])
    file  = open("../results_mpi/primvals_mpi" * string(storePointsAll[1]) * "_" * string(rank) * "_" * string(getConfig()["core"]["max_iters"]) * ".txt", "w")
    for (idx, itm) in enumerate(globaldata)
        primtowrite = globaldata[idx].min_q
        for element in primtowrite
            @printf(file,"%0.17f", element)
            @printf(file, " ")
        end
        print(file, "\n")
    end
    close(file)

end
