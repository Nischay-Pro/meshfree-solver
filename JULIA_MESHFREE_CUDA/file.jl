function returnFileLength(file_name::String)
    data1 = read(file_name, String)
    splitdata = split(data1, "\n")
    return length(splitdata) - 1
end

function readFile(file_name::String, globaldata, table, defprimal)
    data1 = read(file_name, String)
    splitdata = @view split(data1, "\n")[1:end-1]
    # print(splitdata[1:3])
    for (idx, itm) in enumerate(splitdata)
        itmdata = split(itm, " ")
        temp =  Point(parse(Int,itmdata[1]),
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
        table[idx] = globaldata[idx].localID
        # globaldataLocalID[idx] = parse(Int,itmdata[1])
        # globaldataCommon[1:8, idx] =
        # globaldataDq[idx] = Array{Array{Float64,1},2}(undef, 2 , 4)
        # if count == 0
        #     print(temp)
        # end

        # if parse(Int, itmdata[6]) == 1
        #     wallpts += 1
        #     push!(wallptsidx, parse(Int,itmdata[1]))
        # elseif parse(Int, itmdata[6]) == 2
        #     Interiorpts += 1
        #     push!(Interiorptsidx, parse(Int,itmdata[1]))
        # elseif parse(Int,itmdata[1]) == 3
        #     outerpts += 1
        #     push!(outerptsidx, parse(Int,itmdata[1]))
        # end
        # if parse(Int, itmdata[7]) > 0
        #     shapepts +=1
        #     push!(shapeptsidx, parse(Int,itmdata[1]))
        # end
        # if count == 0
        #     print(wallptsidx)
        #     print(Interiorptsidx)
        #     print(outerptsidx)
        # end
        # count = 1
    end
    return nothing
end