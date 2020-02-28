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
        connectivity = zeros(Int32, 20)
        for iter in 9:length(itmdata)
            connectivity[iter-8] = parse(Int32, itmdata[iter])
        end

        globaldata[idx] = Point(idx,
                    parse(Float64,itmdata[1]),
                    parse(Float64, itmdata[2]),
                    parse(Int, itmdata[3]),
                    parse(Int, itmdata[4]),
                    parse(Int8,itmdata[5]),
                    parse(Int8,itmdata[6]),
                    parse(Float64,itmdata[7]),
                    parse(Int8,itmdata[8]),
                    SVector{20}(connectivity),
                    0.0,
                    0.0,
                    SVector{4}(defprimal),
                    SVector{4}([zero(Float64) for iter in 1:4]),
                    SVector{4}([zero(Float64) for iter in 1:4]),
                    zeros(Float64, 4),
                    zeros(Float64, 4), 
                    0.0, 0, 0, 0, 0, 
                    SVector{20}([zero(Int32) for iter in 1:20]), 
                    SVector{20}([zero(Int32) for iter in 1:20]),
                    SVector{20}([zero(Int32) for iter in 1:20]), 
                    SVector{20}([zero(Int32) for iter in 1:20]), 
                    0.0, 
                    zeros(Float64, 4), 
                    zeros(Float64, 4), 
                    SVector{4}([zero(Float64) for iter in 1:4]))

    end
    return nothing
end

function readFileQuadtree(file_name::String, globaldata, defprimal, numPoints)
    data1 = read(file_name, String)
    splitdata = @view split(data1, "\n")[2:end-1]
    # print(splitdata[1:3])
    @showprogress 1 "Computing ReadFile" for (idx, itm) in enumerate(splitdata)
        itmdata = split(itm)
        connectivity = zeros(Int32, 20)
        for iter in 12:length(itmdata)
            connectivity[iter-11] = parse(Int32, itmdata[iter])
        end

        globaldata[idx] = Point(idx,
                    parse(Float64,itmdata[1]),
                    parse(Float64, itmdata[2]),
                    parse(Int, itmdata[3]),
                    parse(Int, itmdata[4]),
                    parse(Int8,itmdata[5]),
                    parse(Int8,itmdata[6]),
                    parse(Float64,itmdata[10]),
                    parse(Int8,itmdata[11]),
                    SVector{20}(connectivity),
                    parse(Float64, itmdata[7]),
                    parse(Float64, itmdata[8]),
                    SVector{4}(defprimal),
                    SVector{4}([zero(Float64) for iter in 1:4]),
                    SVector{4}([zero(Float64) for iter in 1:4]),
                    zeros(Float64, 4),
                    zeros(Float64, 4), 
                    0.0, 0, 0, 0, 0, 
                    SVector{20}([zero(Int32) for iter in 1:20]), 
                    SVector{20}([zero(Int32) for iter in 1:20]),
                    SVector{20}([zero(Int32) for iter in 1:20]), 
                    SVector{20}([zero(Int32) for iter in 1:20]), 
                    0.0, 
                    zeros(Float64, 4), 
                    zeros(Float64, 4), 
                    SVector{4}([zero(Float64) for iter in 1:4]))
    end
return nothing
end