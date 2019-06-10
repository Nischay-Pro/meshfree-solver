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


function convertToArray(targetArray2, originalStruct::Point, idx)
    # targetArray[:, idx] =  vcat([
    #                                         originalStruct.localID ,
    #                                         originalStruct.x ,
    #                                         originalStruct.y ,
    #                                         originalStruct.left ,
    #                                         originalStruct.right ,
    #                                         originalStruct.flag_1 ,
    #                                         originalStruct.flag_2 ,
    #                                         originalStruct.nbhs
    #                                     ],
    #                                     zeros(Int32, 20) , #9
    #                                     [
    #                                         originalStruct.nx , #29
    #                                         originalStruct.ny
    #                                     ] ,
    #                                     originalStruct.prim , #31
    #                                     originalStruct.flux_res , #35
    #                                     zeros(Float64, 4) , #39 q
    #                                     zeros(Float64, 8) , #43 dq
    #                                     [
    #                                         originalStruct.entropy , #51
    #                                         originalStruct.xpos_nbhs ,
    #                                         originalStruct.xneg_nbhs ,
    #                                         originalStruct.ypos_nbhs ,
    #                                         originalStruct.yneg_nbhs
    #                                     ] ,
    #                                     zeros(Int32, 20) , #56 xpos_conn
    #                                     zeros(Int32, 20) , #76 xneg_conn
    #                                     zeros(Int32, 20) , #96 ypos_conn
    #                                     zeros(Int32, 20) , #116 yneg_conn
    #                                     [
    #                                         originalStruct.delta , #136
    #                                         originalStruct.short_distance #137
    #                                     ] ,
    #                                     zeros(Float64, 4) , #138
    #                                     zeros(Float64, 4) , #142
    #                                     zeros(Float64, 28)  #146|150|154|158|162|166|170
    #                                                         #phii| phik |Gi| Gk|sumdelx|sumdely|result)
    #                                     )
    # targetArray[9:8 + originalStruct.nbhs, idx] = originalStruct.conn
    # targetArray[56:55+ originalStruct.xpos_nbhs, idx] = originalStruct.xpos_conn
    # targetArray[76:75+ originalStruct.xneg_nbhs, idx] = originalStruct.xneg_conn
    # targetArray[96:95+ originalStruct.ypos_nbhs, idx] = originalStruct.ypos_conn
    # targetArray[116:115+ originalStruct.yneg_nbhs, idx] = originalStruct.yneg_conn
    targetArray2[1] = originalStruct.xpos_nbhs
    targetArray2[2] = originalStruct.xneg_nbhs
    targetArray2[3] = originalStruct.ypos_nbhs
    targetArray2[4] = originalStruct.yneg_nbhs
    targetArray2[5:4 + originalStruct.nbhs, idx] = originalStruct.conn
    targetArray2[15:14 + originalStruct.xpos_nbhs, idx] = originalStruct.xpos_conn
    targetArray2[25:24 + originalStruct.xneg_nbhs, idx] = originalStruct.xneg_conn
    targetArray2[35:34 + originalStruct.ypos_nbhs, idx] = originalStruct.ypos_conn
    targetArray2[45:44 + originalStruct.yneg_nbhs, idx] = originalStruct.yneg_conn
    return  nothing
end


# function setSmallestPointDistance(self::Point, distance)
#     self.short_distance = distance
# end
