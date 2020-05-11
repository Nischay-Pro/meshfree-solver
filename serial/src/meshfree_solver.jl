function main()

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

    interior::Int64 = configData["point"]["interior"]
    wall::Int64 = configData["point"]["wall"]
    outer::Int64 = configData["point"]["outer"]
    @showprogress 2 "Computing Normals" for idx in 1:numPoints
        placeNormals(globaldata, idx, configData, interior, wall, outer)
    end

    println("Start Connectivity Generation")
    @showprogress 3 "Computing Connectivity" for idx in 1:numPoints
        calculateConnectivity(globaldata, idx)
    end

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

    println(max_iters + 1)
    function run_code(globaldata, configData, res_old, numPoints, main_store, tempdq)
        for i in 1:max_iters
            fpi_solver(i, globaldata, configData, res_old, numPoints, main_store, tempdq)
        end
    end

    function test_code(globaldata, configData, res_old, numPoints, main_store)
        println("Starting warmup function")
        # fpi_solver(1, globaldata, configData,  res_old, numPoints)
        res_old[1] = 0.0
        # Profile.clear_malloc_data()
        # Profile.clear()
        # res_old[1] = 0.0
        # fpi_solver(1, globaldata, configData,  res_old, numPoints)
        # @profile fpi_solver(1, globaldata, configData,  res_old)
        # Profile.print()
        # res_old[1] = 0.0
        println("Starting main function")
        tempdq = zeros(Float64, numPoints, 2, 4)
        # @trace(fpi_solver(1, globaldata, configData,  res_old, main_store, tempdq), maxdepth = 3)
        @timeit to "nest 1" begin
            run_code(globaldata, configData, res_old, numPoints, main_store, tempdq)
        end
        # open("prof.txt", "w") do s
        #     Profile.print(IOContext(s, :displaysize => (24, 500)))
        # end
    end


    test_code(globaldata, configData, res_old, numPoints, main_store)

    open("../results/timer" * string(numPoints) * "_" * string(getConfig()["core"]["max_iters"]) *".txt", "w") do io
        print_timer(io, to)
    end

    # file  = open("../results/primvals" * string(numPoints) * "_" * string(getConfig()["core"]["max_iters"]) * ".txt", "w")
    # for (idx, itm) in enumerate(globaldata)
    #     primtowrite = globaldata[idx].prim
    #     for element in primtowrite
    #         @printf(file,"%0.17f", element)
    #         @printf(file, " ")
    #     end
    #     print(file, "\n")
    # end
    # close(file)
end


    # compute_cl_cd_cm(globaldata, configData, shapeptsidx)
    # println(globaldata[1])
