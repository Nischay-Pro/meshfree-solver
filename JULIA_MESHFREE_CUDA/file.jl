function returnFileLength(file_name::String)
    data1 = read(file_name, String)
    splitdata = split(data1, "\n")
    return length(splitdata) - 1
end

function readFile(file_name::String, globaldata, defprimal, globalDataFixedPoint)
    data1 = read(file_name, String)
    splitdata = @view split(data1, "\n")[1:end-1]
    # print(splitdata[1:3])
    for (idx, itm) in enumerate(splitdata)
        itmdata = split(itm, " ")
        temp =  Point(parse(Int32,itmdata[1]),
                    parse(Float64,itmdata[2]),
                    parse(Float64, itmdata[3]),
                    parse(Int32,itmdata[1]) - 1,
                    parse(Int32,itmdata[1]) + 1,
                    parse(Int8,itmdata[6]),
                    parse(Int8,itmdata[7]),
                    parse(Float64,itmdata[8]),
                    parse(Int8,itmdata[9]),
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

        globalDataFixedPoint[idx] = FixedPoint(parse(Int32,itmdata[1]),
                                            parse(Float64,itmdata[2]),
                                            parse(Float64, itmdata[3]),
                                            parse(Int32,itmdata[1]) - 1,
                                            parse(Int32,itmdata[1]) + 1,
                                            parse(Int8,itmdata[6]),
                                            parse(Int8,itmdata[7]),
                                            parse(Float64,itmdata[8]),
                                            parse(Int8,itmdata[9]),
                                            parse(Float64, itmdata[4]),
                                            parse(Float64, itmdata[5]),
                                            0.0
                                                )

        if parse(Int32, itmdata[1]) == 1
            temp.left = 160
            globalDataFixedPoint[idx] = FixedPoint(parse(Int32,itmdata[1]),
                                            parse(Float64,itmdata[2]),
                                            parse(Float64, itmdata[3]),
                                            temp.left,
                                            parse(Int32,itmdata[1]) + 1,
                                            parse(Int8,itmdata[6]),
                                            parse(Int8,itmdata[7]),
                                            parse(Float64,itmdata[8]),
                                            parse(Int8,itmdata[9]),
                                            parse(Float64, itmdata[4]),
                                            parse(Float64, itmdata[5]),
                                            0.0
                                                )
        elseif parse(Int32, itmdata[1]) == 160
            temp.right = 1
            globalDataFixedPoint[idx] = FixedPoint(parse(Int32,itmdata[1]),
                                            parse(Float64,itmdata[2]),
                                            parse(Float64, itmdata[3]),
                                            parse(Int32,itmdata[1]) - 1,
                                            temp.right,
                                            parse(Int8,itmdata[6]),
                                            parse(Int8,itmdata[7]),
                                            parse(Float64,itmdata[8]),
                                            parse(Int8,itmdata[9]),
                                            parse(Float64, itmdata[4]),
                                            parse(Float64, itmdata[5]),
                                            0.0
                                                )
        else
            globalDataFixedPoint[idx] = FixedPoint(parse(Int32,itmdata[1]),
                                            parse(Float64,itmdata[2]),
                                            parse(Float64, itmdata[3]),
                                            parse(Int32,itmdata[1]) - 1,
                                            parse(Int32,itmdata[1]) + 1,
                                            parse(Int8,itmdata[6]),
                                            parse(Int8,itmdata[7]),
                                            parse(Float64,itmdata[8]),
                                            parse(Int8,itmdata[9]),
                                            parse(Float64, itmdata[4]),
                                            parse(Float64, itmdata[5]),
                                            0.0
                                                )
        end

        if idx % 100000 == 0
            println(idx)
        end

        globaldata[idx] = temp

    end
    return nothing
end