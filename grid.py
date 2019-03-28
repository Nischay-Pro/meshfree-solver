import config
import math
import numpy as np
import numba
from numba import cuda
import os

def main():

    configData = config.getConfig()

    NIMAX = configData["grid"]["ijmax"][0]
    NJMAX = configData["grid"]["ijmax"][1]
    r = configData["grid"]["radiusOuter"]
    dc = configData["grid"]["clockwiseOuter"]
    shift = configData["grid"]["shift"]
    m = configData["grid"]["m"]
    p = configData["grid"]["p"]
    t = configData["grid"]["t"]

    pi = 4 * math.atan(1)
    N = NIMAX+1
    normal = 1.00893041136514
    c1 = 0.2969
    c2 = -0.126
    c3 = -0.3516
    c4 = 0.2843
    c5 = -0.1015

    xc = np.zeros(5125, dtype=np.float64)
    yc = np.zeros(5125, dtype=np.float64)

    Xe = np.zeros(10000000, dtype=np.float64)
    Ye = np.zeros(10000000, dtype=np.float64)

    X = np.zeros((5125, 1921), dtype=np.float64)
    Y = np.zeros((5125, 1921), dtype=np.float64)

    for i in range(1, N + 1):
        t1 = 2 * pi * ((i - 1) / (N - 1))
        xc[i] = r * math.cos(t1)
        if shift:
            xc[i] = xc[i] + 0.5
        yc[i] = r * math.sin(t1)

    xc[N] = xc[1]
    yc[N] = yc[1]

    if not os.path.isdir("grids/generated"):
        os.makedirs("grids/generated", exist_ok=True)

    with open("grids/generated/farfield.dat", "w+") as the_file:
        if not dc:
            for i in range(N):
                the_file.write("%s %s\n" % (xc[i], yc[i]))
        else:
            for i in range(N, 0, -1):
                the_file.write("%s %s\n" % (xc[i], yc[i]))

    for i in range(1, NIMAX + 1):
        theta = 2 * pi * (i - 1) / N
        Xe[i] = 0.5 * (math.cos(theta) + 1) * normal
        xn = Xe[i]
        Ye[i] - 0.5 * t * ((c1 * math.sqrt(xn)) + (c2 * xn) + (c3 * math.pow(xn, 2)) + (c4 * math.pow(xn, 3)) + (c5 * math.pow(xn, 4)))

        if theta < pi:
            Ye[i] = -Ye[i]
        
        if m == 0:
            yn = 0
        elif xn <= p:
            yn = m * normal * (2 * p * xn - xn ** 2) / xn ** 2
        else:
            yn = m * normal * (1 - 2 * p + 2 * p * xn - xn**2)/(1 - p)**2
        Ye[i] = Ye[i] + yn
    
    for i in range(1, NIMAX + 1):

        Xe[i] = Xe[i] / normal
        Ye[i] = Ye[i] / normal


    IMAX= NIMAX + 2
    JMAX = NJMAX

    NPOB = IMAX - 2

    tempdata = open("grids/generated/farfield.dat")
    tempdata = tempdata.read()
    tempdata = tempdata.split("\n")
    for i in range(1, NPOB + 1):
        temp = tempdata[i - 1]
        temp = temp.split(" ")
        X[i, JMAX] = float(temp[0])
        Y[i, JMAX] = float(temp[1])

    X[IMAX - 1, JMAX] = X[1, JMAX]
    Y[IMAX - 1, JMAX] = Y[1, JMAX]
    X[IMAX, JMAX] = X[2, JMAX]
    Y[IMAX, JMAX] = Y[2, JMAX]

    NPIB = IMAX - 2

    for i in range (1, IMAX - 1):
        X[i, 1] = Xe[i]
        Y[i, 1] = Ye[i]

    X[IMAX - 1, 1] = X[1, 1]
    Y[IMAX - 1, 1] = Y[1 ,1]
    X[IMAX, 1] = X[2, 1]
    Y[IMAX, 1] = Y[2, 1]

    IS = 2
    JS = 2
    IE = IMAX - 1
    JE = JMAX - 1

    GG1(X, Y, IMAX, JMAX, IS, IE, JS, JE)


def GG1(X, Y, IMAX, JMAX, IS, IE, JS, JE):
    GG2(X, Y, IMAX, JMAX)
    GG3(X, Y, IS, IE, JS, JE, IMAX, JMAX)

    with open("grids/generated/grid.dat", "w+") as the_file:
        for j in range(1, JMAX + 1):
            for i in range(1, IMAX - 1):
                the_file.write("%s %s %s %s\n" % (X[i, j], Y[i, j], i , j))

    with open("grids/generated/gnu.dat", "w+") as the_file:
        for i in range(1, IMAX - 1):
            for j in range(1, JMAX):
                the_file.write("%s %s\n" % (X[i, j], Y[i, j]))
                the_file.write("%s %s\n" % (X[i, j + 1], Y[i, j + 1]))
                the_file.write("%s %s\n" % (X[i, j], Y[i, j]))
                the_file.write("%s %s\n" % (X[i + 1, j], Y[i + 1, j]))
    
    with open("grids/generated/temp.dat", "w+") as the_file:
        for j in range(1, IMAX - 1):
            the_file.write("%s %s\n" % (X[j, JMAX - 1], Y[j, JMAX - 1]))

"""
SUBROUTINE: GRID GENERATION - 2
FINDS X, Y VALUES AT ALL NODES
USES LAGRANGIAN INTERPOLATION SCHEME

TRANSFINITE LINEAR INTERPOLATION
"""

def GG2(X, Y, IMAX, JMAX):
    for i in range(1, IMAX + 1):
        for j in range(2 , JMAX):
            X[i, j] = X[i, 1] + ((j - 1) / (JMAX - 1)) * (X[i, JMAX] - X[i, 1])
            Y[i, j] = Y[i, 1] + ((j - 1) / (JMAX - 1)) * (Y[i, JMAX] - Y[i, 1])

def GG3(X, Y, IS, IE, JS, JE, IMAX, JMAX):
    P = np.zeros(5125, dtype=np.float64)
    Q = np.zeros(5125, dtype=np.float64)
    APX = np.zeros((5125, 1921), dtype=np.float64)
    AWX = np.zeros((5125, 1921), dtype=np.float64)
    ANX = np.zeros((5125, 1921), dtype=np.float64)
    ASX = np.zeros((5125, 1921), dtype=np.float64)
    XOLD = np.zeros((5125, 1921), dtype=np.float64)
    YOLD = np.zeros((5125, 1921), dtype=np.float64)
    AEX = np.zeros((5125, 1921), dtype=np.float64)
    ZX = np.zeros((5125, 1921), dtype=np.float64)

    res = input("Enter: ")
    if len(res) == 0:
        A, B, C, D = 1, 1, 1, 1
    else:
        res = res.split(" ")
        A = int(res[0])
        B = int(res[1])
        C = int(res[2])
        D = int(res[3])

    K = 1
    L = 1

    for i in range(IS , IE + 1):
        for j in range(JS, JE + 1):
            if i == K:
                P[i] = 0
            else:
                P[i] = (-A) * ((i - K) / abs(i - K)) * math.exp((-B) * abs(i - K))
            if j == L:
                Q[j] = 0
            else:
                Q[j] = (-C) * ((j - L) / abs(j - L)) * math.exp((-D) * abs(j - L))          

    RMSE = 10000
    while RMSE > 0.0001:
        for j in range(1, JMAX + 1):
            X[IMAX, j] = X[2, j]
            Y[IMAX, j] = Y[2, j]
            X[1, j] = X[IMAX - 1, j]
            Y[1, j] = Y[IMAX - 1, j]

        for i in range(IS, IE + 1):
            for j in range(JS, JE + 1):
                XZ = (X[i + 1, j] - X[i - 1, j]) / 2
                XE = (X[i, j + 1] - X[i, j - 1]) / 2
                YZ = (Y[i + 1, j] - Y[i - 1, j]) / 2
                YE = (Y[i, j + 1] - Y[i, j - 1]) / 2

                ALPHA = XE * XE + YE * YE
                GAMA = XZ * XZ + YZ * YZ
                BEETA = XZ * XE + YZ * YE
                JACO = XZ * YE - XE * YZ
                JACO2 = JACO * JACO

                APX[i, j] = (2 / JACO2) * (ALPHA + GAMA)
                AEX[i, j] = (P[i] / 2) + (ALPHA / JACO2)
                AWX[i, j] = (-P[i] / 2) + (ALPHA / JACO2)
                ANX[i, j] = (Q[j] / 2) + (GAMA / JACO2)
                ASX[i, j] = (-Q[j] / 2) + (GAMA / JACO2)
                ZX[i, j] = (-BEETA) / (2 * JACO2)

                if i == j == 40:
                    print(ALPHA, GAMA, BEETA, JACO, JACO2)


        for i in range(IS , IE + 1):
            for j in range(JS, JE + 1):
                XOLD[i, j] = X[i, j]
                YOLD[i, j] = Y[i, j]

        if not cuda.is_available():
            SOLVER(IS, IE, JS, JE, IMAX, JMAX, APX, AEX, AWX, ANX, ASX, ZX, X)
            SOLVER(IS, IE, JS, JE, IMAX, JMAX, APX, AEX, AWX, ANX, ASX, ZX, Y)

        else:
            print("CUDA Mode: Enabled")
            stream = cuda.stream()
            FI = np.zeros((5125, 5125), dtype=np.float64)
            A = np.zeros((5125, 5125), dtype=np.float64)
            B = np.zeros((5125, 5125), dtype=np.float64)
            C = np.zeros((5125, 5125), dtype=np.float64)
            D = np.zeros((5125, 5125), dtype=np.float64)
            blockDim = (32, 32)
            gridDim = (math.ceil(5125 / 32), math.ceil(1921 / 32))
            with stream.auto_synchronize():
                APX_gpu = cuda.to_device(APX, stream)
                AEX_gpu = cuda.to_device(AEX, stream)
                AWX_gpu = cuda.to_device(AWX, stream)
                ANX_gpu = cuda.to_device(ANX, stream)
                ASX_gpu = cuda.to_device(ASX, stream)
                ZX_gpu = cuda.to_device(ZX, stream)
                X_gpu = cuda.to_device(X, stream)
                Y_gpu = cuda.to_device(Y, stream)
                FI_gpu = cuda.to_device(FI, stream)
                A_gpu = cuda.to_device(A, stream)
                B_gpu = cuda.to_device(B, stream)
                C_gpu = cuda.to_device(C, stream)
                D_gpu = cuda.to_device(D, stream)
                print("starting")
                size_T = X.shape

                blockDim = (32, 32)
                gridDim = (math.ceil(size_T[0] / 1024), 1)
                print("X Stage 1")
                for j in range(JS, JE + 1):
                    SOLVER_CUDA_STAGE_2[gridDim, blockDim](IS, IE, APX_gpu, AEX_gpu, AWX_gpu, ANX_gpu, ASX_gpu, ZX_gpu, X_gpu, FI_gpu, A_gpu, B_gpu, C_gpu, D_gpu, j)

                cuda.synchronize()

                print("X Stage 2")
                for j in range(JS, JE + 1):
                    TDMA_CUDA[gridDim, blockDim](A_gpu, B_gpu, C_gpu, D_gpu, IS, IE, FI_gpu, X_gpu, IMAX, False, j)

                cuda.synchronize()

                print("X Stage 3")
                for j in range(JS, JE + 1):
                    SOLVER_CUDA_STAGE_3[gridDim, blockDim](IS, IE, X_gpu, FI_gpu, j)

                cuda.synchronize()

                gridDim = (math.ceil(size_T[1] / 1024), 1)
                print("Y Stage 1")
                for j in range(IS, IE + 1):
                    SOLVER_CUDA_STAGE_2[gridDim, blockDim](JS, JE, APX_gpu, AEX_gpu, AWX_gpu, ANX_gpu, ASX_gpu, ZX_gpu, Y_gpu, FI_gpu, A_gpu, B_gpu, C_gpu, D_gpu, j)

                cuda.synchronize()

                print("Y Stage 2")
                for j in range(IS, IE + 1):
                    TDMA_CUDA[gridDim, blockDim](A_gpu, B_gpu, C_gpu, D_gpu, JS, JE, FI_gpu, Y_gpu, JMAX, True, j)

                cuda.synchronize()

                print("Y Stage 3")
                for j in range(IS, IE + 1):
                    SOLVER_CUDA_STAGE_3[gridDim, blockDim](JS, JE, Y_gpu, FI_gpu, j)

                cuda.synchronize()

                print("Copying back")
                X = X_gpu.copy_to_host()
                Y = Y_gpu.copy_to_host()

                # np.savetxt('test.out', X)

            DIFFX2 = 0
            DIFFY2 = 0
            for i in range(IS, IE + 1):
                for j in range(JS, JE + 1):
                    DIFFX = X[i, j] - XOLD[i, j]
                    DIFFY = Y[i, j] - YOLD[i, j]
                    DIFFX2 = DIFFX**2 + DIFFX2
                    DIFFY2 = DIFFY**2 + DIFFY2
            DIFF2 = DIFFX2
            if DIFFX2 <= DIFFY2:
                DIFF2 = DIFFY2
            RMSE = math.sqrt(DIFF2)
            print(RMSE)
                


def SOLVER(IS, IE, JS, JE, IMAX, JMAX, AP, AE, AW, AN, AS, Z, T):
    FI = np.zeros(5125, dtype=np.float64)
    A = np.zeros(5125, dtype=np.float64)
    B = np.zeros(5125, dtype=np.float64)
    C = np.zeros(5125, dtype=np.float64)
    D = np.zeros(5125, dtype=np.float64)

    for j in range(JS, JE + 1):
        TW = T[1, j]
        TE = T[IMAX, j]

        for i in range(IS, IE + 1):
            A[i] = AP[i, j]
            B[i] = AE[i, j]
            C[i] = AW[i, j]
            D[i] = (AN[i, j] * T[i, j + 1]) + (AS[i, j] * T[i, j - 1]) + (Z[i, j] * (T[i + 1, j + 1] - T[i + 1, j - 1] - T[i - 1, j + 1] + T[i - 1, j - 1]))

        TDMA(A, B, C, D, IS, IE, FI, TW, TE)
        for i in range(IS, IE + 1):
            T[i, j] = FI[i]

    for i in range(IS, IE + 1):
        TN = T[i, JMAX]
        TS = T[i, 1]
        for j in range(JS, JE + 1):
            A[j] = AP[i, j]
            B[j] = AN[i, j]
            C[j] = AS[i, j]
            D[j] = (AE[i, j] * T[i + 1, j]) + (AW[i , j] * T[i - 1, j]) + (Z[i, j] * (T[i + 1, j + 1] - T[i + 1, j - 1] - T[i - 1, j + 1] + T[i - 1, j - 1]))

        TDMA(A, B, C, D, JS, JE, FI, TS, TN)

        for j in range(JS, JE + 1):
            T[i, j] = FI[j]

@cuda.jit()
def SOLVER_CUDA_STAGE_2(IS, IE, AP, AE, AW, AN, AS, Z, T, FI, A, B, C, D, j):
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x
    ty = cuda.threadIdx.y
    by = cuda.blockIdx.y
    bl = cuda.blockDim.y
    i = (bx * bw * by) + (ty * bl) + tx

    if i >= IS and i <= IE:
        A[j, i] = AP[i, j]
        B[j, i] = AE[i, j]
        C[j, i] = AW[i, j]
        D[j, i] = (AN[i, j] * T[i, j + 1]) + (AS[i, j] * T[i, j - 1]) + (Z[i, j] * (T[i + 1, j + 1] - T[i + 1, j - 1] - T[i - 1, j + 1] + T[i - 1, j - 1]))

@cuda.jit()
def SOLVER_CUDA_STAGE_3(IS, IE, T, FI, j):
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x
    ty = cuda.threadIdx.y
    by = cuda.blockIdx.y
    bl = cuda.blockDim.y
    i = (bx * bw * by) + (ty * bl) + tx

    if i >= IS and i <= IE:
        T[i, j] = FI[j, i]

@cuda.jit()
def TDMA_CUDA(A, B, C, D, IS, IE, FI, T, MAX, Y_MODE, CONTROL):

    if Y_MODE == True:
        T1 = T[CONTROL, 1]
        T2 = T[CONTROL, MAX]
    else:
        T1 = T[1, CONTROL]
        T2 = T[MAX, CONTROL]
    
    # AA = numba.cuda.local.array(5125, dtype=numba.float64)
    # BB = numba.cuda.local.array(5125, dtype=numba.float64)
    AA = np.zeros(5125, dtype=np.float64)
    BB = np.zeros(5125, dtype=np.float64)       


    AA[IS] = B[CONTROL, IS] / A[CONTROL, IS]
    BB[IS] = (C[CONTROL, IS] * T1 + D[CONTROL, IS]) / A[CONTROL, IS]

    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x
    ty = cuda.threadIdx.y
    by = cuda.blockIdx.y
    bl = cuda.blockDim.y
    i = (bx * bw * by) + (ty * bl) + tx

    if i >= IS + 1 and i <= IE:
        DR = A[CONTROL, i] - (C[CONTROL, i] * AA[i - 1])
        AA[i] = B[CONTROL, i] / DR
        BB[i] = (D[CONTROL, i] + (BB[i - 1] * C[CONTROL, i])) / DR
    
    FI[CONTROL, IE] = AA[IE] * T2 + BB[IE]

    if i <= IE - 1 and i >= IS:
        FI[CONTROL, i] = (AA[i] * FI[CONTROL, i + 1]) + BB[i]

def TDMA(A, B, C, D, IS, IE, FI, T1, T2):
    AA = np.zeros(5125, dtype=np.float64)
    BB = np.zeros(5125, dtype=np.float64)       

    AA[IS] = B[IS] / A[IS]
    BB[IS] = (C[IS] * T1 + D[IS]) / A[IS]

    for i in range(IS + 1, IE + 1):
        DR = A[i] - (C[i] * AA[i - 1])
        AA[i] = B[i] / DR
        BB[i] = (D[i] + (BB[i - 1] * C[i])) / DR
    
    FI[IE] = AA[IE] * T2 + BB[IE]

    for i in range(IE - 1, IS - 1, -1):
        FI[i] = (AA[i] * FI[i + 1]) + BB[i]
    

if __name__ == "__main__":
    main()