
function main()
    globaldata = Array{Point,1}(undef, 0)
    configData = getConfig()
    wallpts, Interiorpts, outerpts, shapepts = 0,0,0,0
    wallptsidx = Array{Int,1}(undef, 0)
    Interiorptsidx = Array{Int,1}(undef, 0)
    outerptsidx = Array{Int,1}(undef, 0)
    shapeptsidx = Array{Int,1}(undef, 0)
    table = Array{Int,1}(undef, 0)

    file1 = open("partGridNew--160-60")
    data1 = read(file1, String)
    splitdata = split(data1, "\n")
    # print(splitdata[1:3])
    splitdata = splitdata[1:end-1]
    # print(splitdata[1:3])
    defprimal = getInitialPrimitive(configData)

    # count = 0
    for (idx, itm) in enumerate(splitdata)
        itmdata = split(itm, " ")
        temp = Point(parse(Int,itmdata[1]),
                    parse(Float64,itmdata[2]),
                    parse(Float64, itmdata[3]),
                    parse(Int,itmdata[1]) - 1,
                    parse(Int,itmdata[1]) + 1,
                    parse(Int,itmdata[6]),
                    parse(Int,itmdata[7]),
                    parse(Float64,itmdata[8]),
                    parse(Int,itmdata[9]),
                    parse.(Int, itmdata[10:end-1]),
                    parse(Float64, itmdata[4]),
                    parse(Float64, itmdata[5]),
                    copy(defprimal),
                    zeros(Float64, 4),
                    zeros(Float64, 4),
                    Array{Array{Float64,1},1}(undef, 0), 0.0, 0, 0, 0, 0, Array{Int,1}(undef, 0), Array{Int,1}(undef, 0),
                    Array{Int,1}(undef, 0), Array{Int,1}(undef, 0), 0.0, zeros(Float64, 4), zeros(Float64, 4))

        if parse(Int, itmdata[1]) == 1
            temp.left = 160
            temp.right = 2
        end

        if parse(Int, itmdata[1]) == 160
            temp.left = 159
            temp.right = 1
        end

        # print(convert(String, temp))
        # print(globaldata)
        # print("123\n")
        push!(globaldata, temp)
        # if count == 0
        #     print(temp)
        # end

        if parse(Int, itmdata[6]) == 1
            wallpts += 1
            push!(wallptsidx, parse(Int,itmdata[1]))
        elseif parse(Int, itmdata[6]) == 2
            Interiorpts += 1
            push!(Interiorptsidx, parse(Int,itmdata[1]))
        elseif parse(Int,itmdata[1]) == 3
            outerpts += 1
            push!(outerptsidx, parse(Int,itmdata[1]))
        end
        if parse(Int, itmdata[7]) > 0
            shapepts +=1
            push!(shapeptsidx, parse(Int,itmdata[1]))
        end
        push!(table, parse(Int,itmdata[1]))
        # if count == 0
        #     print(wallptsidx)
        #     print(Interiorptsidx)
        #     print(outerptsidx)
        # end
        # count = 1
    end

    # print(wallptsidx)

    for idx in table
        connectivity = calculateConnectivity(globaldata, idx)
        setConnectivity(globaldata[idx], connectivity)
        # smallest_dist(globaldata, idx)
    end


    # print(Int(getConfig()["core"]["max_iters"]) + 1)
    function run_code(globaldata, configData, wallptsidx, outerptsidx, Interiorptsidx, res_old)
        for i in 1:(Int(getConfig()["core"]["max_iters"]))
            fpi_solver(i, globaldata, configData, wallptsidx, outerptsidx, Interiorptsidx, res_old)
        end
    end

    # @trace(fpi_solver(1, globaldata, configData, wallptsidx, outerptsidx, Interiorptsidx, res_old), maxdepth = 3)



    res_old = zeros(Float64, 1)
    function test_code(globaldata, configData, wallptsidx, outerptsidx, Interiorptsidx, res_old)
        fpi_solver(1, globaldata, configData, wallptsidx, outerptsidx, Interiorptsidx, res_old)
        println("Starting process")
        res_old[1] = 0.0
        @timeit to "nest 4" begin
            run_code(globaldata, configData, wallptsidx, outerptsidx, Interiorptsidx, res_old)
        end
    end

    test_code(globaldata, configData, wallptsidx, outerptsidx, Interiorptsidx, res_old)
    # println(to)
    open("timer.txt", "w") do io
        print_timer(io, to)
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
end
