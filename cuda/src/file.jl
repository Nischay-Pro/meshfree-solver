function returnFileLength(file_name::String)
    data1 = read(file_name, String)
    splitdata = split(data1, "\n")
    return length(splitdata) - 2
end

function returnHDF5FileLength(file_name::String)
    file_id = h5open(file_name, "r")
    group_id = file_id["/1"]
    dset = read(open_attribute(group_id, "total"))
    max_points = dset[1]
    close(group_id)
    close(file_id)
    return max_points
end

function readHDF5File(file_name::String, globaldata, defprimal, globalDataRest, numPoints)
    file_id = h5open(file_name, "r")
    group_id = file_id["/1"]
    dataset_id = open_dataset(group_id, "local")
    itm = read(dataset_id)
    for idx in 1:numPoints
    # println(itm[:, 1])
        temp =  Point(itm[1, idx],
            itm[2, idx],
            itm[3, idx],
            itm[7, idx],
            itm[8, idx],
            itm[10, idx],
            itm[11, idx],
            itm[6, idx],
            itm[12, idx],
            itm[13:12 + Int(itm[12, idx]), idx],
            itm[4, idx],
            itm[6, idx],
            0.0,
            0,
            0,
            0,
            0,
            Array{Int,1}(undef, 0),
            Array{Int,1}(undef, 0),
            Array{Int,1}(undef, 0),
            Array{Int,1}(undef, 0),
            0.0)

        globaldata[idx] = temp
        globalDataRest[idx, 1:4] = copy(defprimal)
    end
    close(dataset_id)
    close(group_id)
    close(file_id)
    return nothing
end 
