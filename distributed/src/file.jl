function returnFileLength(file_name::String)
    data1 = read(file_name, String)
    splitdata = split(data1, "\n")
    return length(splitdata) - 2
end

function createGlobalLocalMapIndex(index_holder, folder_name::String)
    index_flag = 0
    # println("Reading multiple files")
    @sync for iter in 1:length(workers())
        if iter - 1 < 10
            filename = folder_name * "/" * "partGrid000" * string(iter-1)
        elseif iter - 1 < 100
            filename = folder_name * "/" * "partGrid00" * string(iter-1)
        elseif iter - 1 < 1000
            filename = folder_name * "/" * "partGrid0" * string(iter-1)
        else
            filename = folder_name * "/" * "partGrid" * string(iter-1)
        end
        # println(filename)
        idx = 1
        local_point_count = 0
        fp = open(filename)
        file_iter = eachline(fp)
        for splitdata in file_iter
            if idx == 1
                itmdata = split(splitdata)
                local_point_count = parse(Int,itmdata[3])
                break
            end
        end
        close(fp)
        index_flag += local_point_count
        @spawnat iter+1 index_holder[:L][1] = index_flag
    end
end

function readGhostFile(folder_name::String, ghost_holder, dist_globaldata, p)
    # println(ghost_holder)
    iter = p - 1
    if iter - 1 < 10
        filename = folder_name * "/" * "partGrid000" * string(iter-1)
    elseif iter - 1 < 100
        filename = folder_name * "/" * "partGrid00" * string(iter-1)
    elseif iter - 1 < 1000
        filename = folder_name * "/" * "partGrid0" * string(iter-1)
    else
        filename = folder_name * "/" * "partGrid" * string(iter-1)
    end
    # println(filename)
    idx = 1
    local_point_count = 0
    ghost_point_count = 0
    ghost_holder[1] = Dict{Int64,Point}()
    fp = open(filename)
    file_iter = eachline(fp)

    for splitdata in file_iter
        if idx == local_point_count + ghost_point_count + 2
           break 
        end
        
        if idx == 1
            itmdata = split(splitdata)
            local_point_count = parse(Int,itmdata[3])
            ghost_point_count = parse(Int,itmdata[4])
        end
        if idx >= local_point_count + 2
            itmdata = split(splitdata)
            globalidx = parse(Int,itmdata[1])
            ghost_holder[1][idx-1] = dist_globaldata[globalidx]
        end
        idx+=1
    end
    close(fp)
    return nothing
end

function readDistribuedFile(folder_name::String, defprimal, p, global_local_map_index)
    # println("Reading multiple files")
    # println(folder_name)
    iter = p - 1
    if iter - 1 < 10
        filename = folder_name * "/" * "partGrid000" * string(iter-1)
    elseif iter - 1 < 100
        filename = folder_name * "/" * "partGrid00" * string(iter-1)
    elseif iter - 1 < 1000
        filename = folder_name * "/" * "partGrid0" * string(iter-1)
    else
        filename = folder_name * "/" * "partGrid" * string(iter-1)
    end
    println(filename)
    data = read(filename, String)
    splitdata = @view split(data, "\n")[1:end-1]
    itmdata = split(splitdata[1])
    local_point_count = parse(Int,itmdata[3])
    ghost_point_count = parse(Int,itmdata[4])
    local_points_holder = Array{Point,1}(undef, local_point_count)
    for (idx, itm) in enumerate(splitdata)
        if idx == 1
            continue
        elseif idx <= local_point_count + 1
            itmdata = split(itm)
            globalID = global_local_map_index[(parse(Float64,itmdata[2]), parse(Float64, itmdata[3]))]
            local_points_holder[idx-1] = Point(parse(Int,itmdata[1]),
                parse(Float64, itmdata[2]),
                parse(Float64, itmdata[3]),
                parse(Int, itmdata[4]),
                parse(Int, itmdata[5]),
                parse(Int8,itmdata[6]),
                parse(Int8,itmdata[7]),
                parse(Float64,itmdata[8]),
                parse(Int8,itmdata[9]),
                parse.(Int, itmdata[10:end]),
                0.0,
                0.0,
                copy(defprimal),
                zeros(Float64, 4),
                zeros(Float64, 4),
                zeros(Float64, 4), zeros(Float64, 4), zeros(Float64, 4), zeros(Float64, 4), 0.0, 0, 0, 0, 0, Array{Int32,1}(undef, 0), Array{Int32,1}(undef, 0),
                Array{Int32,1}(undef, 0), Array{Int32,1}(undef, 0), 0.0, zeros(Float64, 4), zeros(Float64, 4), zeros(Float64, 4),
                globalID)
        end
    end
    return local_points_holder
end

function readDistribuedFileQuadtree(folder_name::String, defprimal, p, index_holder)
    # println(folder_name)
    iter = p - 1
    if iter - 1 < 10
        filename = folder_name * "/" * "partGrid000" * string(iter-1)
    elseif iter - 1 < 100
        filename = folder_name * "/" * "partGrid00" * string(iter-1)
    elseif iter - 1 < 1000
        filename = folder_name * "/" * "partGrid0" * string(iter-1)
    else
        filename = folder_name * "/" * "partGrid" * string(iter-1)
    end
    # println(filename)
    idx = 1
    local_point_count = 0
    ghost_point_count = 0
    local_points_holder = []

    store_index = index_holder[:L][1]

    fp = open(filename)
    file_iter = eachline(fp)

    for splitdata in file_iter
        if idx == 1
            itmdata = split(splitdata)
            local_point_count = parse(Int,itmdata[3])
            ghost_point_count = parse(Int,itmdata[4])
            local_points_holder = Array{Point,1}(undef, local_point_count)
            store_index -= local_point_count
        elseif idx <= local_point_count + 1
            itmdata = split(splitdata)
            globalID = store_index + idx - 1
            local_points_holder[idx-1] = Point(parse(Int,itmdata[1]),
                parse(Float64, itmdata[2]),
                parse(Float64, itmdata[3]),
                parse(Int, itmdata[4]),
                parse(Int, itmdata[5]),
                parse(Int8,itmdata[6]),
                parse(Int8,itmdata[7]),
                parse(Float64,itmdata[11]),
                parse(Int8,itmdata[12]),
                parse.(Int, itmdata[13:end]),
                parse(Float64, itmdata[7]),
                parse(Float64, itmdata[8]),
                copy(defprimal),
                SVector{4}([zero(Float64) for iter in 1:4]),
                zeros(Float64, 4),
                SVector{4}([zero(Float64) for iter in 1:4]), 
                SVector{4}([zero(Float64) for iter in 1:4]), 
                SVector{4}([zero(Float64) for iter in 1:4]), 
                SVector{4}([zero(Float64) for iter in 1:4]), 
                0.0, 0, 0, 0, 0,
                SVector{25}([zero(Int32) for iter in 1:25]), 
                SVector{25}([zero(Int32) for iter in 1:25]),
                SVector{25}([zero(Int32) for iter in 1:25]), 
                SVector{25}([zero(Int32) for iter in 1:25]), 
                0.0, zeros(Float64, 4), zeros(Float64, 4), zeros(Float64, 4),
                globalID)
        else
            break
        end
        idx+=1
    end
    close(fp)
    return local_points_holder
end

function readDistribuedFileQ(folder_name::String, defprimal, p)

    # println(folder_name)
    iter = p - 1
    if iter - 1 < 10
        filename = folder_name * "/" * "partGrid000" * string(iter-1)
    elseif iter - 1 < 100
        filename = folder_name * "/" * "partGrid00" * string(iter-1)
    elseif iter - 1 < 1000
        filename = folder_name * "/" * "partGrid0" * string(iter-1)
    else
        filename = folder_name * "/" * "partGrid" * string(iter-1)
    end
    # println(filename)
    idx = 1
    local_point_count = 0
    ghost_point_count = 0
    local_points_holder = []

    fp = open(filename)
    file_iter = eachline(fp)

    for splitdata in file_iter
        if idx == 1
            itmdata = split(splitdata)
            local_point_count = parse(Int,itmdata[3])
            ghost_point_count = parse(Int,itmdata[4])
            break
        end
    end
    close(fp)

    local_points_holder = [TempQ(SVector{4}([zero(Float64) for iter in 1:4])) for idx in 1:local_point_count]

    return local_points_holder
end

function readDistribuedFileQPack(folder_name::String, defprimal, p)

    # println(folder_name)
    iter = p - 1
    if iter - 1 < 10
        filename = folder_name * "/" * "partGrid000" * string(iter-1)
    elseif iter - 1 < 100
        filename = folder_name * "/" * "partGrid00" * string(iter-1)
    elseif iter - 1 < 1000
        filename = folder_name * "/" * "partGrid0" * string(iter-1)
    else
        filename = folder_name * "/" * "partGrid" * string(iter-1)
    end
    idx = 1
    local_point_count = 0
    ghost_point_count = 0
    local_points_holder = []

    fp = open(filename)
    file_iter = eachline(fp)

    for splitdata in file_iter
        if idx == 1
            itmdata = split(splitdata)
            local_point_count = parse(Int,itmdata[3])
            ghost_point_count = parse(Int,itmdata[4])
            break
        end
    end
    close(fp)
    local_points_holder = [TempQPack(SVector{4}([zero(Float64) for iter in 1:4]), 
        SVector{4}([zero(Float64) for iter in 1:4]), 
        SVector{4}([zero(Float64) for iter in 1:4]), 
        SVector{4}([zero(Float64) for iter in 1:4]), 
        SVector{4}([zero(Float64) for iter in 1:4])) for idx in 1:local_point_count]

    return local_points_holder
end

function returnKeys(loc_ghost_holder)
    keysIter = keys(loc_ghost_holder[1])
    keysLength = length(keysIter)
    keyHolder = zeros(Int64, keysLength)
    idx = 1
    for item in keysIter
        keyHolder[idx] = item
        idx += 1
    end
    return keyHolder
end