function main()

    configData = getConfig()
    max_iters = parse(Int, ARGS[2])
    file_name = string(ARGS[1])
    format = configData["format"]["type"]
    numPoints = returnFileLength(file_name)

    println(numPoints)
    globaldata = Array{Point,1}(undef, numPoints)
    res_old = zeros(Float64, 1)
    main_store = zeros(Float64, 52)

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

    globaldata = StructArray(globaldata)

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
end


    # compute_cl_cd_cm(globaldata, configData, shapeptsidx)

    # println(IOContext(stdout, :compact => false), globaldata[1].q)
    # println(IOContext(stdout, :compact => false), globaldata[1].dq)
    # println(IOContext(stdout, :compact => false), globaldata[100].q)
    # println(IOContext(stdout, :compact => false), globaldata[100].dq)
    # println(IOContext(stdout, :compact => false), globaldata[1000].q)
    # println(IOContext(stdout, :compact => false), globaldata[1000].dq)
    # println()
    # println(IOContext(stdout, :compact => false), globaldata[1].flux_res)
    # println(IOContext(stdout, :compact => false), globaldata[100].flux_res)
    # println(IOContext(stdout, :compact => false), globaldata[1000].flux_res)
    # println()
    # println(IOContext(stdout, :compact => false), globaldata[1].delta)
    # println(IOContext(stdout, :compact => false), globaldata[100].delta)
    # println(IOContext(stdout, :compact => false), globaldata[1000].delta)
    # println()
    # println(IOContext(stdout, :compact => false), globaldata[1].prim)
    # println(IOContext(stdout, :compact => false), globaldata[100].prim)
    # println(IOContext(stdout, :compact => false), globaldata[1000].prim)
    # println(IOContext(stdout, :compact => false), globaldata[100].ypos_conn)
    # println(IOContext(stdout, :compact => false), globaldata[100].yneg_conn)
    # println(globaldata[1])
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
