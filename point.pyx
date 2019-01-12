# cython: profile=True
# cython: binding=True
# cython: linetrace=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1
import numpy as np 
cimport numpy as np

DTYPE = np.float64
DTYPE_LONG = np.long

ctypedef np.float64_t DTYPE_t
ctypedef np.long_t DTYPE_LONG_t

cdef class Point:
    cdef int localID,left,right,flag_1,flag_2,nbhs,xpos_nbhs,xneg_nbhs,ypos_nbhs,yneg_nbhs
    cdef double x,y,nx,ny,delta, entropy
    cdef np.ndarray conn,prim,flux_res,q,dq,xpos_conn,xneg_conn,ypos_conn,yneg_conn

    def __cinit__(self, int localID, double x, double y, int left, int right, int flag_1, int flag_2, int nbhs, np.ndarray conn, double nx, double ny, np.ndarray prim, np.ndarray flux_res, np.ndarray q, np.ndarray dq, double entropy, int xpos_nbhs, int xneg_nbhs, int ypos_nbhs, int yneg_nbhs, np.ndarray[DTYPE_LONG_t] xpos_conn, np.ndarray[DTYPE_LONG_t] xneg_conn, np.ndarray[DTYPE_LONG_t] ypos_conn, np.ndarray[DTYPE_LONG_t] yneg_conn, double delta):
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

    def getleft(self):
        return self.left

    def getright(self):
        return self.right
    
    def getx(self):
        return self.x
    
    def gety(self):
        return self.y
    
    def get_flag_1(self):
        return self.flag_1

    def get_flag_2(self):
        return self.flag_2

    def getlocalID(self):
        return self.localID

    def get_conn(self):
        return self.conn

    def getnx(self):
        return self.nx
    
    def getny(self):
        return self.ny

    def getprim(self):
        return self.prim

    def setprim(self, prim):
        self.prim = prim

    def get_flux_res(self):
        return self.flux_res

    def set_flux_res(self, flux):
        self.flux_res = flux

    def getq(self):
        return self.q

    def setq(self, q):
        self.q = q

    def getdq(self):
        return self.dq

    def setdq(self, dq):
        self.dq = dq

    def getEntropy(self):
        return self.entropy

    def getDelta(self):
        return self.delta

    def setDelta(self, delta):
        self.delta = delta

    def get_xpos_conn(self):
        return self.xpos_conn

    def get_xneg_conn(self):
        return self.xneg_conn

    def get_ypos_conn(self):
        return self.ypos_conn

    def get_yneg_conn(self):
        return self.yneg_conn