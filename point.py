class Point:
    def __init__(self, localID, x, y, left, right, flag_1, flag_2, nbhs, conn, nx, ny, prim, flux_res, q, dq, entropy, xpos_nbhs, xneg_nbhs, ypos_nbhs, yneg_nbhs, xpos_conn, xneg_conn, ypos_conn, yneg_conn, delta):
        self.localID = localID
        self.x = x
        self.y = y
        self.left = left
        self.right = right
        self.flag_1 = flag_1
        self.flag_2 = flag_2
        self.nbhs = nbhs
        self.conn = conn
        self.nx = nx
        self.ny = ny
        # Size 4 (Pressure, vx, vy, density) x numberpts
        self.prim = prim
        self.flux_res = flux_res
        # Size 4 (Pressure, vx, vy, density) x numberpts
        self.q = q
        # Size 2(x,y) 4(Pressure, vx, vy, density) numberpts
        self.dq = dq
        self.entropy = entropy
        self.xpos_nbhs = xpos_nbhs
        self.xneg_nbhs = xneg_nbhs
        self.ypos_nbhs = ypos_nbhs
        self.yneg_nbhs = yneg_nbhs
        self.xpos_conn = xpos_conn
        self.xneg_conn = xneg_conn
        self.ypos_conn = ypos_conn
        self.yneg_conn = yneg_conn
        self.delta = delta
