function returnFileLength(file_name::String)
    data1 = read(file_name, String)
    splitdata = split(data1, "\n")
    return length(splitdata) - 1
end

function readFile(file_name::String, globaldata, table, defprimal, wallptsidx, outerptsidx, Interiorptsidx, shapeptsidx,
        wallpts, Interiorpts, outerpts, shapepts)
    data1 = read(file_name, String)
    splitdata = @view split(data1, "\n")[1:end-1]
    # print(splitdata[1:3])
    @showprogress 1 "Computing ReadFile" for (idx, itm) in enumerate(splitdata)
        itmdata = split(itm, " ")
        globaldata[idx] = Point(parse(Int,itmdata[1]),
                    parse(Float64,itmdata[2]),
                    parse(Float64, itmdata[3]),
                    parse(Int,itmdata[1]) - 1,
                    parse(Int,itmdata[1]) + 1,
                    parse(Int8,itmdata[6]),
                    parse(Int8,itmdata[7]),
                    parse(Float64,itmdata[8]),
                    parse(Int8,itmdata[9]),
                    parse.(Int32, itmdata[10:end-1]),
                    parse(Float64, itmdata[4]),
                    parse(Float64, itmdata[5]),
                    copy(defprimal),
                    zeros(Float64, 4),
                    zeros(Float64, 4),
                    Array{Array{Float64,1},1}(undef, 0), 0.0, 0, 0, 0, 0, Array{Int32,1}(undef, 0), Array{Int32,1}(undef, 0),
                    Array{Int32,1}(undef, 0), Array{Int32,1}(undef, 0), 0.0, zeros(Float64, 4), zeros(Float64, 4))

        # println(temp)
        # if idx % 100000 == 0
        #     println(idx)
        # end

        if globaldata[idx].localID == 1
            globaldata[idx].left = 160
            # globaldata[idx].right = 2
        end

        if globaldata[idx].localID == 160
            # globaldata[idx].left = 5119
            globaldata[idx].right = 1
        end
        if globaldata[idx].flag_1 == 1
            wallpts += 1
            push!(wallptsidx, globaldata[idx].localID)
        elseif globaldata[idx].flag_1 == 2
            Interiorpts += 1
            push!(Interiorptsidx, globaldata[idx].localID)
        elseif globaldata[idx].flag_1 == 3
            outerpts += 1
            push!(outerptsidx, globaldata[idx].localID)
        end
        if globaldata[idx].flag_2 > 0
            shapepts +=1
            push!(shapeptsidx, globaldata[idx].localID)
        end
        table[idx] = globaldata[idx].localID
    end
    return nothing
end