mutable struct Point
    localID::Int
    x::Float64
    y::Float64
    left::Int
    right::Int
    flag_1::Int
    flag_2::Int
    nbhs
    conn
    nx::Float64
    ny::Float64
    # Size 4 (Pressure, vx, vy, density) x numberpts
    prim
    flux_res
    # Size 4 (Pressure, vx, vy, density) x numberpts
    q
    # Size 2(x,y) 4(Pressure, vx, vy, density) numberpts
    dq
    entropy
    xpos_nbhs
    xneg_nbhs
    ypos_nbhs
    yneg_nbhs
    xpos_conn
    xneg_conn
    ypos_conn
    yneg_conn
    delta
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
    self.xpos_nbhs = size(conn[1])
    self.xneg_conn = conn[2]
    self.xneg_nbhs = size(conn[2])
    self.ypos_conn = conn[3]
    self.ypos_nbhs = size(conn[3])
    self.yneg_conn = conn[4]
    self.yneg_nbhs = size(conn[4])
end
