mutable struct Point
    localID::Int64 #1
    x::Float64
    y::Float64
    left::Int64
    right::Int64
    flag_1::Int64
    flag_2::Int64
    nbhs::Int64 #8
    conn::Array{Int64,1} #9
    nx::Float64 #10
    ny::Float64
    # Size 4 (Pressure, vx, vy, density) x numberpts
    prim::Array{Float64,1} #12
    flux_res::Array{Float64,1} #16
    # Size 4 (Pressure, vx, vy, density) x numberpts
    q::Array{Float64,1} #20
    # Size 2(x,y) 4(Pressure, vx, vy, density) numberpts
    dq::Array{Array{Float64,1},1} #24
    entropy::Float64 #32
    xpos_nbhs::Int64
    xneg_nbhs::Int64
    ypos_nbhs::Int64
    yneg_nbhs::Int64
    xpos_conn::Array{Int64,1}
    xneg_conn::Array{Int64,1}
    ypos_conn::Array{Int64,1}
    yneg_conn::Array{Int64,1}
    delta::Float64 #
    short_distance::Float64
    max_q::Array{Float64,1}
    min_q::Array{Float64,1}
end

# point = Point(locaslID, x, y, left, right, flag_1, flag_2, nbhs, conn, nx, ny, prim, flux_res, q, dq, entropy, xpos_nbhs, xneg_nbhs, ypos_nbhs, yneg_nbhs, xpos_conn, xneg_conn, ypos_conn, yneg_conn, delta)

function setNormals(self::Point, n)
    self.nx = n[1]
    self.ny = n[2]
end

function getxy(self::Point)
    return (self.x, self.y)
end

function setConnectivity(self::Point, conn)
    self.xpos_conn = conn[1]
    self.xpos_nbhs = length(conn[1])
    self.xneg_conn = conn[2]
    self.xneg_nbhs = length(conn[2])
    self.ypos_conn = conn[3]
    self.ypos_nbhs = length(conn[3])
    self.yneg_conn = conn[4]
    self.yneg_nbhs = length(conn[4])
end


function convertToArray(targetArray, originalStruct::Point, idx)
    targetArray[:, idx] =  vcat([
                                            originalStruct.localID ,
                                            originalStruct.x ,
                                            originalStruct.y ,
                                            originalStruct.left ,
                                            originalStruct.right ,
                                            originalStruct.flag_1 ,
                                            originalStruct.flag_2 ,
                                            originalStruct.nbhs
                                        ],
                                        zeros(Float64, 20) , #9
                                        [
                                            originalStruct.nx , #29
                                            originalStruct.ny
                                        ] ,
                                        originalStruct.prim , #31
                                        originalStruct.flux_res , #35
                                        zeros(Float64, 8) , #39
                                        [
                                            originalStruct.entropy , #47
                                            originalStruct.xpos_nbhs ,
                                            originalStruct.xneg_nbhs ,
                                            originalStruct.ypos_nbhs ,
                                            originalStruct.yneg_nbhs
                                        ] ,
                                        zeros(Float64, 20) , #52
                                        zeros(Float64, 20) , #72
                                        zeros(Float64, 20) , #92
                                        zeros(Float64, 20) , #112
                                        [
                                            originalStruct.delta , #132
                                            originalStruct.short_distance #133
                                        ] ,
                                        zeros(Float64, 4) , #134
                                        zeros(Float64, 4) #138
                                        )
    targetArray[9:8 + originalStruct.nbhs, idx] = originalStruct.conn
end


# function setSmallestPointDistance(self::Point, distance)
#     self.short_distance = distance
# end