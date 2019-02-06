function main()

    globaldata = []

    configData = getConfig()

    wallpts, Interiorpts, outerpts = 0,0,0
    wallptsidx = Array{Int,1}(undef, 0)
    Interiorptsidx = Array{Int,1}(undef, 0)
    outerptsidx = Array{Int,1}(undef, 0)
    table = Array{Int,1}(undef, 0)

    file1 = open("partGridNew")
    data1 = read(file1, String)
    splitdata = split(data1, "\n")
    splitdata = splitdata[1:end-1]

    defprimal = getInitialPrimitive2(configData)

    for (idx, itm) in enumerate(splitdata)
        itmdata = split(itm, " ")
        temp = Point(parse.(Int,itmdata[1]), parse.(Float64,itmdata[2]), parse.(Float64, itmdata[3]), 1, 1, parse.(Int,itmdata[6]), parse.(Int,itmdata[7]), parse.(Int,itmdata[8]), parse.(Int,itmdata[9:end]), parse.(Float64, itmdata[4]), parse.(Float64, itmdata[5]), defprimal[idx], nothing, zeros(Float64, 4), nothing, nothing, 0, 0, 0, 0, nothing, nothing, nothing, nothing, nothing, nothing)
        # print(convert(String, temp))
        # print(globaldata)
        # print("123\n")
        globaldata  = push!(globaldata, temp)
        # print(globaldata[1])
        if convert(Int, parse.(Float64, itmdata[6])) == 1
            wallpts += 1
            push!(wallptsidx, convert(Int, parse.(Float64,itmdata[1])))
        elseif convert(Int, parse.(Float64, itmdata[6])) == 2
            Interiorpts += 1
            push!(Interiorptsidx, convert(Int, parse.(Float64,itmdata[1])))
        elseif convert(Int, parse.(Float64, itmdata[6])) == 3
            outerpts += 1
            push!(outerptsidx, convert(Int, parse.(Float64, itmdata[1])))
        end
        push!(table, convert(Int, parse.(Float64, itmdata[1])))
    end

    for idx in table
        connectivity = calculateConnectivity(globaldata, idx)
        # if idx == 1
        #     print("\n\n====1>", connectivity)
        # end
        setConnectivity(globaldata[idx], connectivity)
        globaldata[idx].short_distance = smallest_dist(globaldata, idx)
    end

    res_old = 0
    # print(Int(getConfig()["core"]["max_iters"]) + 1)
    for i in 1:(Int(getConfig()["core"]["max_iters"]))
        # print(i)
        # println(globaldata[77])
        fpi_solver(i, globaldata, configData, wallptsidx, outerptsidx, Interiorptsidx, res_old)
        # println(globaldata[77])
    end

    file  = open("primvals.txt", "w")
    for (idx, itm) in enumerate(globaldata)
        primtowrite = globaldata[idx].prim
        for element in primtowrite
            print(file, element)
            print(file, " ")
        end
        print(file, "\n")
    end
    close(file)
end
