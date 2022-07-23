class Point:
    def __init__(self, localID, x, y, left, right, flag_1, flag_2, nbhs, conn, nx, ny, prim, min_dist):
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
        self.flux_res = None
        # Size 4 (Pressure, vx, vy, density) x numberpts
        self.q = None
        # Size 2(x,y) 4(Pressure, vx, vy, density) numberpts
        self.dq = None
        self.entropy = None
        self.xpos_nbhs = None
        self.xneg_nbhs = None
        self.ypos_nbhs = None
        self.yneg_nbhs = None
        self.xpos_conn = None
        self.xneg_conn = None
        self.ypos_conn = None
        self.yneg_conn = None
        self.delta = None
        self.min_dist = min_dist
        self.qtdepth = None

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

    def checkConnectivity(self):
        return not self.nbhs == len(self.conn)