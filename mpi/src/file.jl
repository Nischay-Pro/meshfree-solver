function specificCoreFile(file_name::String)
    file_name *= "partGrid"
    for iter in 1:4 - length(string(MPI.Comm_rank(MPI.COMM_WORLD)))
        file_name *= "0"
    end
    file_name *= string(MPI.Comm_rank(MPI.COMM_WORLD))
    return file_name
end


function returnFileLength(file_name::String)
    data1 = read(file_name, String)
    splitdata = split(data1, "\n")
    return length(splitdata) - 2
end

function readFile(file_name::String, globaldata, defprimal)
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
                    SVector{4}([zero(Float64) for iter in 1:4]),
                    SVector{4}([zero(Float64) for iter in 1:4]), 
                    0.0, 0, 0, 0, 0, 
                    SVector{20}([zero(Int32) for iter in 1:20]), 
                    SVector{20}([zero(Int32) for iter in 1:20]),
                    SVector{20}([zero(Int32) for iter in 1:20]), 
                    SVector{20}([zero(Int32) for iter in 1:20]), 
                    0.0, 
                    SVector{4}([zero(Float64) for iter in 1:4]), 
                    SVector{4}([zero(Float64) for iter in 1:4]), 
                    SVector{4}([zero(Float64) for iter in 1:4]))

    end
    return nothing
end

function readFileQuadtree(file_name::String, globaldata, defprimal)
    data1 = read(file_name, String)
    splitdata = @view split(data1, "\n")[2:end-1]
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
                    SVector{4}([zero(Float64) for iter in 1:4]), 
                    SVector{4}([zero(Float64) for iter in 1:4]), 
                    SVector{4}([zero(Float64) for iter in 1:4]))
    end
    return nothing
end

function readFileMPIQuadtree(file_name::String, globaldata, defprimal, localPoints, ghostPoints)
    data1 = read(file_name, String)
    metadata = split(data1, "\n")[1]
    metadata = split(metadata)
    localPoints = parse(Int, metadata[3])
    ghostPoints = parse(Int, metadata[4])
    partitionGhostLimit = parse(Int, metadata[5])
    # println(localPoints, " Test ", ghostPoints)
    splitdata = @view split(data1, "\n")[2:end-1]
    
    @showprogress 1 "Computing ReadFile" for (idx, itm) in enumerate(splitdata)
        itmdata = split(itm)

        if idx <= localPoints
            connectivity = zeros(Int32, 20)
            for iter in 12:length(itmdata)
                connectivity[iter-11] = parse(Int32, itmdata[iter])
            end

            globaldata[idx] = Point(idx,
                parse(Int, itmdata[1]),
                parse(Float64,itmdata[2]),
                parse(Float64, itmdata[3]),
                parse(Int, itmdata[4]),
                parse(Int, itmdata[5]),
                parse(Int8,itmdata[6]),
                parse(Int8,itmdata[7]),
                parse(Float64,itmdata[11]),
                parse(Int8,itmdata[12]),
                SVector{20}(connectivity),
                parse(Float64, itmdata[8]),
                parse(Float64, itmdata[9]),
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
                SVector{4}([zero(Float64) for iter in 1:4]), 
                SVector{4}([zero(Float64) for iter in 1:4]), 
                SVector{4}([zero(Float64) for iter in 1:4]))
        else
            globaldata[idx] = Point(idx,
                parse(Int, itmdata[3]),
                parse(Float64,itmdata[4]),
                parse(Float64, itmdata[5]),
                parse(Int, itmdata[1]),
                parse(Int, itmdata[2]),
                0,
                0,
                parse(Float64,itmdata[6]),
                0,
                SVector{20}(zeros(20)),
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
                SVector{4}([zero(Float64) for iter in 1:4]), 
                SVector{4}([zero(Float64) for iter in 1:4]), 
                SVector{4}([zero(Float64) for iter in 1:4]))
        end    
    end
    # Catch for file reading error
    if globaldata[end].localID != localPoints + ghostPoints
        println(globaldata[end].localID)
        error(" MPI file reading mismatch ")
    end
    return localPoints, ghostPoints, partitionGhostLimit
end