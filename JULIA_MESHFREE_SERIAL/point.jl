mutable struct Point
    localID::Int64
    x::Float64
    y::Float64
    left::Int64
    right::Int64
    flag_1::Int64
    flag_2::Int64
    short_distance::Float64
    nbhs::Int64
    conn::Array{Int64,1}
    nx::Float64
    ny::Float64
    # Size 4 (Pressure, vx, vy, density) x numberpts
    prim::Array{Float64,1}
    flux_res::Array{Float64,1}
    # Size 4 (Pressure, vx, vy, density) x numberpts
    q::Array{Float64,1}
    # Size 2(x,y) 4(Pressure, vx, vy, density) numberpts
    dq::Array{Array{Float64,1},1}
    entropy::Float64
    xpos_nbhs::Int64
    xneg_nbhs::Int64
    ypos_nbhs::Int64
    yneg_nbhs::Int64
    xpos_conn::Array{Int64,1}
    xneg_conn::Array{Int64,1}
    ypos_conn::Array{Int64,1}
    yneg_conn::Array{Int64,1}
    delta::Float64
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

# function setSmallestPointDistance(self::Point, distance)
#     self.short_distance = distance
# end
