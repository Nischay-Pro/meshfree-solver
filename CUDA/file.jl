function returnFileLength(file_name::String)
    data1 = read(file_name, String)
    splitdata = split(data1, "\n")
    return length(splitdata) - 2
end

function readFile(file_name::String, globaldata, defprimal, globalDataRest, numPoints)
    data1 = read(file_name, String)
    splitdata = @view split(data1, "\n")[1:end-1]
    # print(splitdata[1:3])
    @showprogress 1 "Computing ReadFile" for (idx, itm) in enumerate(splitdata)
        itmdata = split(itm, " ")
        # println(itmdata)
        temp =  Point(idx,
                    parse(Float64,itmdata[1]),
                    parse(Float64, itmdata[2]),
                    parse(Int, itmdata[3]),
                    parse(Int, itmdata[4]),
                    parse(Int8,itmdata[5]),
                    parse(Int8,itmdata[6]),
                    parse(Float64,itmdata[7]),
                    parse(Int8,itmdata[8]),
                    parse.(Int, itmdata[9:end-1]),
                    0.0,
                    0.0,
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

        # if idx % 100000 == 0
        #     println(idx)
        # end

        globaldata[idx] = temp
        globalDataRest[1:4, idx] = copy(defprimal)
    end
    return nothing
end

function readFileExtra(file_name::String, globaldata, defprimal, globalDataRest, numPoints)
    data1 = read(file_name, String)
    splitdata = @view split(data1, "\n")[1:end-1]
    # print(splitdata[1:3])
    @showprogress 1 "Computing ReadFile" for (idx, itm) in enumerate(splitdata)
        itmdata = split(itm, " ")
        # println(itmdata)
        if idx == 1
            continue
        else
            idx = idx - 1
        end
        temp =  Point(idx,
                    parse(Float64,itmdata[1]),
                    parse(Float64, itmdata[2]),
                    parse(Int, itmdata[3]),
                    parse(Int, itmdata[4]),
                    parse(Int8,itmdata[5]),
                    parse(Int8,itmdata[6]),
                    parse(Float64,itmdata[7]),
                    parse(Int8,itmdata[8]),
                    parse.(Int, itmdata[9:end-1]),
                    0.0,
                    0.0,
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

        globaldata[idx] = temp
        globalDataRest[1:4, idx] = copy(defprimal)
    end
    return nothing
end

function readFileExtra2(file_name::String, globaldata, defprimal, globalDataRest, numPoints)
    data1 = read(file_name, String)
    splitdata = @view split(data1, "\n")[2:end-1]
    # print(splitdata[1:3])
    @showprogress 1 "Computing ReadFile" for (idx, itm) in enumerate(splitdata)
        itmdata = split(itm, " ")
        temp =  Point(idx,
                    parse(Float64,itmdata[1]),
                    parse(Float64, itmdata[2]),
                    parse(Int32,itmdata[3]),
                    parse(Int32,itmdata[4]),
                    parse(Int8,itmdata[5]),
                    parse(Int8,itmdata[6]),
                    parse(Float64,itmdata[10]),
                    parse(Int8,itmdata[11]),
                    parse.(Int, itmdata[12:end-1]),
                    parse(Float64, itmdata[7]),
                    parse(Float64, itmdata[8]),
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

        # globalDataFixedPoint[idx] = FixedPoint(idx,
        #                                     parse(Float64,itmdata[1]),
        #                                     parse(Float64, itmdata[2]),
        #                                     parse(Int32,itmdata[3]),
        #                                     parse(Int32,itmdata[4]),
        #                                     parse(Int8,itmdata[5]),
        #                                     parse(Int8,itmdata[6]),
        #                                     parse(Float64,itmdata[10]),
        #                                     parse(Int8,itmdata[11]),
        #                                     parse(Float64, itmdata[7]),
        #                                     parse(Float64, itmdata[8]),
        #                                     0.0
                                                # )

        globaldata[idx] = temp
        globalDataRest[1:4, idx] = copy(defprimal)
    end
    return nothing
end