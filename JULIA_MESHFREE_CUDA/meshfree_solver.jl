
function main()
    configData = getConfig()
    # globaldataCommon = zeros(Float64, 38, getConfig()["core"]["points"])
    file_name = string(ARGS[1])
    numPoints = returnFileLength(file_name)
    println("Number of points ", numPoints)
    # globaldataDq = [Array{Array{Float64,1},2}(undef,2,4) for iterating in 1:getConfig()["core"]["points"]]
    # gpuGlobaldataDq = [CuArray{CuArray{Float64,1},2}(undef,2,4) for iterating in 1:getConfig()["core"]["points"]]
    # gpu_globaldata = CuArray{Point,1}(undef, getConfig()["core"]["points"])
    globaldata = Array{Point,1}(undef, numPoints)
    # globalDataCommon = zeros(Float64, 173, numPoints)
    globalDataRest = zeros(Float64, 33, numPoints)
    globalDataFixedPoint = Array{FixedPoint,1}(undef, numPoints)
    globalDataConn = zeros(Int32, 55, numPoints)

    println(sizeof(globalDataRest))
    println(sizeof(globalDataFixedPoint))
    println(sizeof(globalDataConn))

    # table = Array{Int,1}(undef, numPoints)
    defprimal = getInitialPrimitive(configData)
    # wallpts, Interiorpts, outerpts, shapepts = 0,0,0,0
    # wallptsidx = Array{Int,1}(undef, 0)
    # Interiorptsidx = Array{Int,1}(undef, 0)
    # outerptsidx = Array{Int,1}(undef, 0)
    # shapeptsidx = Array{Int,1}(undef, 0)
    println("Start Read")
    readFile(file_name::String, globaldata, defprimal, globalDataFixedPoint, globalDataRest, numPoints)
    # file1 = open("partGridNew--160-60")
    # data1 = read(file1, String)
    # splitdata = split(data1, "\n")
    # print(splitdata[1:3])
    # splitdata = splitdata[1:end-1]
    # print(splitdata[1:3])

    println("Passing to CPU Globaldata")
    # count = 0

    format = configData["format"]["type"]
    # print(wallptsidx)

    if format == 1
        interior = configData["point"]["interior"]
        wall = configData["point"]["wall"]
        outer = configData["point"]["outer"]
        @showprogress 2 "Computing Connectivity" for idx in 1:numPoints
            placeNormals(globaldata, idx, configData, interior, wall, outer)
            convertToFixedArray(globalDataFixedPoint, globaldata[idx], idx, numPoints)
        end
    end

    println("Start table sorting")
    @showprogress 3 "Computing Table" for idx in 1:numPoints
        connectivity = calculateConnectivity(globaldata, idx, configData)
        setConnectivity(globaldata[idx], connectivity)
        # smallest_dist(globaldata, idx)
        convertToNeighbourArray(globalDataConn, globaldata[idx], idx)

    end



    # return
    # typeof(globalDataConn[1])
    # print(globaldata[1].dq)
    # println(typeof(globaldata))
    # println(globaldata[2762])
    # println(globaldata[2763])
    # globaldata_copy = deepcopy(globaldata)
    # gpu_globaldata[1:2000] = CuArray(globaldata[1:2000])
    # println(typeof(gpuGlobaldataDq))
    # gpuGlobaldataLocalID = CuArray(globaldataLocalID)
    gpuSumResSqr = CuArrays.zeros(Float32, numPoints)
    gpuSumResSqrOutput = CuArrays.zeros(Float32, numPoints)
    println("Passing to GPU Globaldata")
    # gpuGlobalDataCommon = CuArray(globalDataCommon)
    gpuConfigData = CuArray([
                            getConfig()["core"]["points"],#1
                            getConfig()["core"]["cfl"],
                            getConfig()["core"]["max_iters"],
                            getConfig()["core"]["mach"],
                            getConfig()["core"]["aoa"],#5
                            getConfig()["core"]["power"],
                            getConfig()["core"]["limiter_flag"],
                            getConfig()["core"]["vl_const"],
                            getConfig()["core"]["initial_conditions_flag"],
                            getConfig()["core"]["interior_points_normal_flag"],#10
                            getConfig()["core"]["shapes"],
                            getConfig()["core"]["rho_inf"],
                            getConfig()["core"]["pr_inf"],
                            getConfig()["core"]["threadsperblock"],
                            getConfig()["core"]["gamma"],#15
                            getConfig()["core"]["clcd_flag"],
                            getConfig()["point"]["wall"],
                            getConfig()["point"]["interior"],
                            getConfig()["point"]["outer"]
                        ])
    gpuGlobalDataFixedPoint = CuArray(globalDataFixedPoint)
    gpuGlobalDataRest = CuArray(globalDataRest)
    gpuGlobalDataConn = CuArray(globalDataConn)
    println("GPU ConfigGlobaldata Finished")
    # gpuGlobaldataDq = CuArray(globaldataDq)
    # println()
    # println(isbitstype(globaldata) == true)
    # d_globaldata_out = similar(d_globaldata)
    # d_globaldata2 = CuArray(globaldata[2001:4000])
    # d_globaldata = CuArray(globaldata_copy)
    # testarray = Array(d_globaldata)
    # globaldata[1:256] = Array(globaldata1)
    # println(sizeof(d_globaldata))
    # println(sizeof(globaldata[2762]))
    # println(sizeof(globaldata[2763]))
    res_old = 0
    # print(Int(getConfig()["core"]["max_iters"]) + 1)
    # for i in 1:(Int(getConfig()["core"]["max_iters"]))
    # println(globaldata[3])
    # fpi_solver(1, globaldata, configData, wallptsidx, outerptsidx, Interiorptsidx, res_old)
    # end
    # for i in 1:(Int(getConfig()["core"]["max_iters"]))
    #     fpi_solver(i, globaldata, configData, wallptsidx, outerptsidx, Interiorptsidx, res_old)
    # end

    # len = 10^7
    # input = ones(Int32, len)
    # output = similar(input)
    # gpu_input = CuArray(input)
    # gpu_output = CuArray(output)

    threadsperblock = Int(getConfig()["core"]["threadsperblock"])
    threadsperblock = parse(Int , ARGS[2])
    blockspergrid = Int(ceil(numPoints/threadsperblock))

    function test_code(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, gpuConfigData, gpuSumResSqr, gpuSumResSqrOutput, threadsperblock,blockspergrid, numPoints)
        println("Starting warmup function")
        fpi_solver_cuda(1, gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, gpuConfigData, gpuSumResSqr, gpuSumResSqrOutput, threadsperblock, blockspergrid, numPoints)
        res_old = 0
        println("Starting main function")
        @timeit to "nest 1" begin
            CuArrays.@sync begin fpi_solver_cuda(Int(getConfig()["core"]["max_iters"]), gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, gpuConfigData, gpuSumResSqr, gpuSumResSqrOutput,
                threadsperblock,blockspergrid, numPoints)
            end
        end
    end

    test_code(gpuGlobalDataConn, gpuGlobalDataFixedPoint, gpuGlobalDataRest, gpuConfigData, gpuSumResSqr, gpuSumResSqrOutput, threadsperblock,blockspergrid, numPoints)

    open("results/timer_cuda" * string(numPoints) * "_" * string(threadsperblock) *".txt", "w") do io
        print_timer(io, to)
    end
    globalDataPrim = Array(gpuGlobalDataRest)

    # println()
    # println(globalDataCommon[:,1])
    # println()
    # println(globalDataCommon1[:,200])
    # println()
    # println(globalDataCommon1[:,end])

    # compute_cl_cd_cm_kernel(globalDataCommon, configData, shapeptsidx)

    # file  = open("primvals.txt", "w")
    # for (idx, itm) in enumerate(globaldata)
    #     primtowrite = globaldata[idx].prim
    #     for element in primtowrite
    #         @printf(file,"%0.17f", element)
    #         @printf(file, " ")
    #     end
    #     print(file, "\n")
    # end
    # close(file)

    println("Writing cuda file")
    file  = open("results/primvals_cuda" * string(numPoints) * ".txt", "w")
    for idx in 1:numPoints
        primtowrite = globalDataPrim[1:4, idx]
        for element in primtowrite
            @printf(file,"%0.17f", element)
            @printf(file, " ")
        end
        print(file, "\n")
    end
    close(file)

end
