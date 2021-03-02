function main()
    configData = getConfig()
    file_name = string(ARGS[1])
    format = configData["format"]["type"]

    numPoints = returnHDF5FileLength(file_name)
    println("Number of points ", numPoints)

    globaldata = Array{Point,1}(undef, numPoints)

    globalDataRest = zeros(Float64, numPoints, 37)
    globalDataConn = zeros(Int32, numPoints, 20)
    globalDataConnSection = zeros(Int8, numPoints, 60)
    globalDataFauxFixed = zeros(Float64, 6 * numPoints)
    println(sizeof(globalDataRest))
    println(sizeof(globalDataFauxFixed))
    println(sizeof(globalDataConn))
    println(sizeof(globalDataConnSection))

    defprimal = getInitialPrimitive(configData)

    readHDF5File(file_name, globaldata, defprimal, globalDataRest, numPoints)

    println("Passing to CPU Globaldata")

    interior = configData["point"]["interior"]
    wall = configData["point"]["wall"]
    outer = configData["point"]["outer"]
    @showprogress 2 "Computing Connectivity" for idx in 1:numPoints
        placeNormals(globaldata, idx, configData, interior, wall, outer)
        convertToFauxArray(globalDataFauxFixed, globaldata[idx], idx, numPoints)
    end

    println("Start table sorting")
    @showprogress 3 "Computing Table" for idx in 1:numPoints
        connectivity = calculateConnectivity(globaldata, idx, configData)
        setConnectivity(globaldata[idx], connectivity)
        # smallest_dist(globaldata, idx)
        convertToNeighbourArray(globalDataConn, globalDataConnSection, globaldata[idx], idx)
    end


    dev = CuDevice(0)
    # CuContext(dev) do ctx

    println("Running on ", dev)

    gpuSumResSqr = CUDA.zeros(Float64, numPoints)
    gpuSumResSqrOutput = CUDA.zeros(Float64, numPoints)
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
    gpuGlobalDataRest = CuArray(globalDataRest)
    gpuGlobalDataConn = CuArray(globalDataConn)
    gpuGlobalDataConnSection = CuArray(globalDataConnSection)
    gpuGlobalDataFauxFixed = CuArray(globalDataFauxFixed)
    println("GPU ConfigGlobaldata Finished")

    res_old = 0

    threadsperblock = Int(configData["core"]["threadsperblock"])
    threadsperblock = parse(Int , ARGS[2])
    blockspergrid = Int(ceil(numPoints/threadsperblock))

    function test_code(gpuGlobalDataConn, gpuGlobalDataConnSection, gpuGlobalDataFauxFixed, gpuGlobalDataRest, gpuConfigData, gpuSumResSqr, gpuSumResSqrOutput, threadsperblock, blockspergrid, numPoints)
        println("Starting warmup function")
        # fpi_solver_cuda(1, gpuGlobalDataConn, gpuGlobalDataConnSection, gpuGlobalDataFauxFixed, gpuGlobalDataFauxFixed, gpuGlobalDataRest, gpuConfigData, gpuSumResSqr, gpuSumResSqrOutput, threadsperblock, blockspergrid, numPoints)
        res_old = 0
        println("Starting main function")
        @timeit to "nest 1" begin
            CUDA.@sync begin fpi_solver_cuda(Int(configData["core"]["max_iters"]), configData["core"]["innerloop"], gpuGlobalDataConn, gpuGlobalDataConnSection, gpuGlobalDataFauxFixed, gpuGlobalDataRest, gpuConfigData, gpuSumResSqr, gpuSumResSqrOutput,
                threadsperblock, blockspergrid, numPoints)
            end
        end
    end

    test_code(gpuGlobalDataConn, gpuGlobalDataConnSection, gpuGlobalDataFauxFixed, gpuGlobalDataRest, gpuConfigData, gpuSumResSqr, gpuSumResSqrOutput, threadsperblock,blockspergrid, numPoints)

    open("../results/timer_cuda" * string(numPoints) * "_" * string(threadsperblock) * "_" * string(configData["core"]["max_iters"]) *
            "_" * string(configData["core"]["innerloop"]) *".txt", "w") do io
        print_timer(io, to)
    end
    # globalDataPrim = Array(gpuGlobalDataRest)


    # println()
    # println(globalDataCommon[:,1])
    # println()
    # println(globalDataCommon1[:,200])
    # println()
    # println(globalDataCommon1[:,end])

    # compute_cl_cd_cm(globalDataFixedPoint, globalDataPrim, configData)

    # stagnation_pressure(globalDataPrim, numPoints, configData)

    # println("Writing cuda file")
    # file  = open("../results/primvals_cuda" * string(numPoints) * "_" * string(threadsperblock) * "_" * string(configData["core"]["max_iters"]) *
    #         "_" * string(configData["core"]["innerloop"]) * ".txt", "w")
    # for idx in 1:numPoints
    #    primtowrite = globalDataPrim[idx, 1:4]
    #    for element in primtowrite
    #        @printf(file,"%0.17f", element)
    #        @printf(file, " ")
    #    end
    #    print(file, "\n")
    # end
    # close(file)
    # end
end
