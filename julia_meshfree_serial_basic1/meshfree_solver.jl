function main()
    globaldata = Array{Point,1}(undef, 0)
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
        temp = Point(parse(Int,itmdata[1]), parse(Float64,itmdata[2]), parse(Float64, itmdata[3]), 1, 1, parse(Int,itmdata[6]), parse(Int,itmdata[7]), parse(Int,itmdata[8]), parse.(Int, itmdata[9:end]), parse(Float64, itmdata[4]), parse(Float64, itmdata[5]), defprimal[idx], zeros(Float64, 4), zeros(Float64, 4), Array{Array{Float64,1},1}(undef, 0), 0.0, 0, 0, 0, 0, Array{Int,1}(undef, 0), Array{Int,1}(undef, 0), Array{Int,1}(undef, 0), Array{Int,1}(undef, 0), 0.0, 0.0)
        # print(convert(String, temp))
        # print(globaldata)
        # print("123\n")
        push!(globaldata, temp)
        # print(globaldata[1])
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
        push!(table, parse(Int,itmdata[1]))
    end

    for idx in table
        connectivity = calculateConnectivity(globaldata, idx)
        setConnectivity(globaldata[idx], connectivity)
        smallest_dist(globaldata, idx)
    end

    res_old = 0
    # print(Int(getConfig()["core"]["max_iters"]) + 1)
    for i in 1:(Int(getConfig()["core"]["max_iters"]))
        fpi_solver(i, globaldata, configData, wallptsidx, outerptsidx, Interiorptsidx, res_old)
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
