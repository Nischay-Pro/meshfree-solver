# point = Point(locaslID, x, y, left, right, flag_1, flag_2, nbhs, conn, nx, ny, prim, flux_res, q, dq, entropy, xpos_nbhs, xneg_nbhs, ypos_nbhs, yneg_nbhs, xpos_conn, xneg_conn, ypos_conn, yneg_conn, delta)

function setNormals(self::Point, n)
    self.nx = n[1]
    self.ny = n[2]
    return nothing
end

function getxy(self::Point)
    return (self.x, self.y)
end

@inline function setConnectivity(self::Point, conn)
    self.xpos_conn = conn[1]
    self.xpos_nbhs = length(conn[1])
    self.xneg_conn = conn[2]
    self.xneg_nbhs = length(conn[2])
    self.ypos_conn = conn[3]
    self.ypos_nbhs = length(conn[3])
    self.yneg_conn = conn[4]
    self.yneg_nbhs = length(conn[4])
    return nothing
end

function convertToFixedArray(targetArray1, originalStruct::Point, idx, numPoints)
    if idx == 1
            targetArray1[idx] = FixedPoint(originalStruct.localID,
                                            originalStruct.x,
                                            originalStruct.y,
                                            numPoints,
                                            originalStruct.localID + 1,
                                            originalStruct.flag_1,
                                            originalStruct.flag_2,
                                            originalStruct.short_distance,
                                            originalStruct.nbhs,
                                            originalStruct.nx,
                                            originalStruct.ny,
                                            0.0
                                                )
    elseif idx == numPoints
            targetArray1[idx] = FixedPoint(originalStruct.localID,
                                            originalStruct.x,
                                            originalStruct.y,
                                            originalStruct.localID - 1,
                                            1,
                                            originalStruct.flag_1,
                                            originalStruct.flag_2,
                                            originalStruct.short_distance,
                                            originalStruct.nbhs,
                                            originalStruct.nx,
                                            originalStruct.ny,
                                            0.0
                                                )
    else
        targetArray1[idx] = FixedPoint(originalStruct.localID,
                                            originalStruct.x,
                                            originalStruct.y,
                                            originalStruct.localID - 1,
                                            originalStruct.localID + 1,
                                            originalStruct.flag_1,
                                            originalStruct.flag_2,
                                            originalStruct.short_distance,
                                            originalStruct.nbhs,
                                            originalStruct.nx,
                                            originalStruct.ny,
                                            0.0
                                                )
    end
end

function convertToNeighbourArray(targetArray1, targetArray2, originalStruct::Point, idx)
    targetArray1[idx, 1:0 + originalStruct.nbhs] = originalStruct.conn
    # targetArray1[idx, 25:24 + originalStruct.xpos_nbhs] = originalStruct.xpos_conn
    # targetArray1[idx, 40:39 + originalStruct.xneg_nbhs] = originalStruct.xneg_conn
    # targetArray1[idx, 55:54 + originalStruct.ypos_nbhs] = originalStruct.ypos_conn
    # targetArray1[idx, 70:69 + originalStruct.yneg_nbhs] = originalStruct.yneg_conn
    for (inner_idx, itm) in enumerate(originalStruct.xpos_conn)
        targetArray2[idx, inner_idx] = findfirst(isequal(itm), originalStruct.conn)
    end
    for (inner_idx, itm) in enumerate(originalStruct.xneg_conn)
        targetArray2[idx, 15 + inner_idx] = findfirst(isequal(itm), originalStruct.conn)
    end
    for (inner_idx, itm) in enumerate(originalStruct.ypos_conn)
        targetArray2[idx, 30 + inner_idx] = findfirst(isequal(itm), originalStruct.conn)
    end
    for (inner_idx, itm) in enumerate(originalStruct.yneg_conn)
        targetArray2[idx, 45 + inner_idx] = findfirst(isequal(itm), originalStruct.conn)
    end
    return nothing
end

function convertToFauxArray(targetArray1, originalStruct::Point, idx, numPoints)
    targetArray1[idx] = originalStruct.x
    targetArray1[idx + numPoints] = originalStruct.y
    targetArray1[idx + 2 * numPoints] = originalStruct.nx
    targetArray1[idx + 3 * numPoints] = originalStruct.ny
    targetArray1[idx + 4 * numPoints] = originalStruct.flag_1
    targetArray1[idx + 5 * numPoints] = originalStruct.short_distance
    return nothing
end

# function setSmallestPointDistance(self::Point, distance)
#     self.short_distance = distance
# end
