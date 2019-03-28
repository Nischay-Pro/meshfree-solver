class Point:
    def __init__(self, localID, x, y, left, right, flag_1, flag_2, nbhs, conn, nx, ny, prim, flux_res, q, dq, entropy, xpos_nbhs, xneg_nbhs, ypos_nbhs, yneg_nbhs, xpos_conn, xneg_conn, ypos_conn, yneg_conn, delta, foreign=False, foreign_core=0, globalID=0, maxq=0, minq=0, ds=0):
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
        self.foreign = foreign
        self.foreign_core = foreign_core
        self.globalID = globalID
        self.maxq = maxq
        self.minq = minq
        self.ds = ds
    

    def setNormals(self, n):
        self.nx = n[0]
        self.ny = n[1]

    def getxy(self):
        return (self.x, self.y)

    def setConnectivity(self, conn):
        self.xpos_conn = conn[0]
        self.xpos_nbhs = len(conn[0])

        self.xneg_conn = conn[1]
        self.xneg_nbhs = len(conn[1])

        self.ypos_conn = conn[2]
        self.ypos_nbhs = len(conn[2])

        self.yneg_conn = conn[3]
        self.yneg_nbhs = len(conn[3])

    def setPrimitive(self, prim):
        # self.prim = prim
        None