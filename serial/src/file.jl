function returnFileLength(file_name::String)
    data1 = read(file_name, String)
    splitdata = split(data1, "\n")
    return length(splitdata) - 2
end

function readFile(file_name::String, globaldata, defprimal, numPoints)
    data1 = read(file_name, String)
    splitdata = @view split(data1, "\n")[2:end-1]
    # print(splitdata[1:3])
    @showprogress 1 "Computing ReadFile" for (idx, itm) in enumerate(splitdata)
        itmdata = split(itm)
        globaldata[idx] = Point(idx,
                    parse(Float64,itmdata[1]),
                    parse(Float64, itmdata[2]),
                    parse(Int, itmdata[3]),
                    parse(Int, itmdata[4]),
                    parse(Int8,itmdata[5]),
                    parse(Int8,itmdata[6]),
                    parse(Float64,itmdata[7]),
                    parse(Int8,itmdata[8]),
                    parse.(Int, itmdata[9:end]),
                    0.0,
                    0.0,
                    copy(defprimal),
                    zeros(Float64, 4),
                    zeros(Float64, 4),
                    zeros(Float64, 4),
                    zeros(Float64, 4), 
                    0.0, 0, 0, 0, 0, 
                    zeros(Int32, 20), 
                    zeros(Int32, 20),
                    zeros(Int32, 20), 
                    zeros(Int32, 20), 
                    0.0, 
                    zeros(Float64, 4), 
                    zeros(Float64, 4), 
                    zeros(Float64, 4))

    end
    return nothing
end

function readFileQuadtree(file_name::String, globaldata, defprimal, numPoints)
    data1 = read(file_name, String)
    splitdata = @view split(data1, "\n")[2:end-1]
    # print(splitdata[1:3])
    @showprogress 1 "Computing ReadFile" for (idx, itm) in enumerate(splitdata)
    itmdata = split(itm)
    globaldata[idx] = Point(idx,
                parse(Float64,itmdata[1]),
                parse(Float64, itmdata[2]),
                parse(Int, itmdata[3]),
                parse(Int, itmdata[4]),
                parse(Int8,itmdata[5]),
                parse(Int8,itmdata[6]),
                parse(Float64,itmdata[10]),
                parse(Int8,itmdata[11]),
                parse.(Int, itmdata[12:end]),
                parse(Float64, itmdata[7]),
                parse(Float64, itmdata[8]),
                copy(defprimal),
                zeros(Float64, 4),
                zeros(Float64, 4),
                zeros(Float64, 4),
                zeros(Float64, 4), 
                0.0, 0, 0, 0, 0, 
                zeros(Int32, 20), 
                zeros(Int32, 20),
                zeros(Int32, 20), 
                zeros(Int32, 20), 
                0.0, 
                zeros(Float64, 4), 
                zeros(Float64, 4), 
                zeros(Float64, 4))
end
return nothing
end