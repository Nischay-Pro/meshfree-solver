
function main()
    globaldata = Array{Point,1}(undef, getConfig()["core"]["points"])
    globaldataCommon = zeros(Float64, 141, getConfig()["core"]["points"])
    # globaldataCommon = zeros(Float64, 38, getConfig()["core"]["points"])

    # globaldataDq = [Array{Array{Float64,1},2}(undef,2,4) for iterating in 1:getConfig()["core"]["points"]]
    # gpuGlobaldataDq = [CuArray{CuArray{Float64,1},2}(undef,2,4) for iterating in 1:getConfig()["core"]["points"]]
    # gpu_globaldata = CuArray{Point,1}(undef, getConfig()["core"]["points"])
    configData = getConfig()
    wallpts, Interiorpts, outerpts, shapepts = 0,0,0,0
    wallptsidx = Array{Int,1}(undef, 0)
    Interiorptsidx = Array{Int,1}(undef, 0)
    outerptsidx = Array{Int,1}(undef, 0)
    shapeptsidx = Array{Int,1}(undef, 0)
    table = Array{Int,1}(undef, 0)

    file1 = open("partGridNew")
    data1 = read(file1, String)
    splitdata = split(data1, "\n")
    # print(splitdata[1:3])
    splitdata = splitdata[1:end-1]
    # print(splitdata[1:3])
    defprimal = getInitialPrimitive(configData)

    # count = 0
    for (idx, itm) in enumerate(splitdata)
        itmdata = split(itm, " ")
        temp =     Point(parse(Int,itmdata[1]),
                        parse(Float64,itmdata[2]),
                        parse(Float64, itmdata[3]),
                        parse(Int,itmdata[1]) - 1,
                        parse(Int,itmdata[1]) + 1,
                        parse(Int,itmdata[6]),
                        parse(Int,itmdata[7]),
                        parse(Int,itmdata[8]),
                        parse.(Int, itmdata[9:end]),
                        parse(Float64, itmdata[4]),
                        parse(Float64, itmdata[5]),
                        copy(defprimal),
                        zeros(Float64, 4),
                        zeros(Float64, 4),
                        Array{Array{Float64,1},1}(undef, 0),
                        0.0,
                        0,
                        0,
                        0,
                        0,
                        Array{Int,1}(undef, 0),
                        Array{Int,1}(undef, 0),
                        Array{Int,1}(undef, 0),
                        Array{Int,1}(undef, 0),
                        0.0,
                        0.0,
                        zeros(Float64, 4),
                        zeros(Float64, 4))

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
        globaldata[idx] = temp
        # globaldataLocalID[idx] = parse(Int,itmdata[1])
        # globaldataCommon[1:8, idx] =
        # globaldataDq[idx] = Array{Array{Float64,1},2}(undef, 2 , 4)
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
        smallest_dist(globaldata, idx)
        convertToArray(globaldataCommon, globaldata[idx], idx)
    end
    # print(globaldata[1].dq)
    # println(typeof(globaldata))
    # println(globaldata[2762])
    # println(globaldata[2763])
    # globaldata_copy = deepcopy(globaldata)
    # gpu_globaldata[1:2000] = CuArray(globaldata[1:2000])
    # println(typeof(gpuGlobaldataDq))
    # gpuGlobaldataLocalID = CuArray(globaldataLocalID)
    gpuGlobaldataCommon = CuArray(globaldataCommon)
    globaldataCommon1 = Array(gpuGlobaldataCommon)
    println(globaldataCommon1[:,1])
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
    for i in 1:(Int(getConfig()["core"]["max_iters"]))
        fpi_solver(i, globaldata, configData, wallptsidx, outerptsidx, Interiorptsidx, res_old)
    end
    compute_cl_cd_cm(globaldata, configData, shapeptsidx)

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

    file  = open("primvals.txt", "w")
    for (idx, itm) in enumerate(globaldata)
        primtowrite = globaldata[idx].prim
        for element in primtowrite
            @printf(file,"%0.17f", element)
            @printf(file, " ")
        end
        print(file, "\n")
    end
    close(file)
end
