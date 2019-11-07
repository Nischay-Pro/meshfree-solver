import math
import core
import numpy as np

def compute_cl_cd_cm(x, y, nx_gpu, ny_gpu, left_gpu, right_gpu, prim, flag_2_gpu, configData, wallindices):

    rho_inf = configData["core"]["rho_inf"]
    Mach = configData["core"]["mach"]
    pr_inf = configData["core"]["pr_inf"]
    shapes = getGeometryCount(flag_2_gpu)
    theta = core.calculateTheta(configData)

    temp = 0.5*rho_inf*Mach*Mach

    H = [0 for i in range(shapes)]
    V = [0 for i in range(shapes)]
    pitch_mom = [0 for i in range(shapes)]

    Cl = [0 for i in range(shapes)]
    Cd = [0 for i in range(shapes)]
    Cm = [0 for i in range(shapes)]

    with open('cp_file', 'w+') as the_file:

        for itm in wallindices:
            left = left_gpu[itm]
            right = right_gpu[itm]
            lx = x[left]
            ly = y[left]
            rx = x[right]
            ry = y[right]
            mx = x[itm]
            my = y[itm]

            ds1 = (mx - lx)**2 + (my - ly)**2
            ds1 = math.sqrt(ds1)

            ds2 = (rx - mx)**2 + (ry - my)**2
            ds2 = math.sqrt(ds2)

            ds = 0.5*(ds1 + ds2)


            nx = nx_gpu[itm]
            ny = ny_gpu[itm]

            cp = prim[itm][3] - pr_inf
            cp = -cp/temp

            flag_2 = flag_2_gpu[itm] - 1
            the_file.write("{} {} {}\n".format(flag_2, mx, cp))

            H[flag_2] += cp * nx * ds
            V[flag_2] += cp * ny * ds

            pitch_mom[flag_2] += (-cp * ny * ds * (mx - 0.25)) + (cp * nx * ds * my)

    V = np.array(V)
    H = np.array(H)

    Cl = V*math.cos(theta) - H*math.sin(theta)
    Cd = H*math.cos(theta) + V*math.sin(theta)
    Cm = pitch_mom

    if configData["core"]["clcd_flag"]:
        print("Cl:",Cl)
        print("Cd:",Cd)

def getGeometryCount(flag_2):
    maxFlag = 0
    for itm in flag_2[1:]:
        if maxFlag < itm:
            maxFlag = itm
    return maxFlag

def printPrimalOutput(x, y, flag_1, prim, conn, configData, iterations, residue, globaldata):
    sensor = computeAdaptSensor(x, y, prim, conn)
    gamma = configData["core"]["gamma"]
    pr_inf = configData["core"]["pr_inf"]
    entropy = computeEntropy(prim, pr_inf, gamma)
    with open('output.dat', "w+") as the_file:
        the_file.write("{} {} {}".format(len(x) - 1, iterations, residue))
        for i in range(1, len(x)):
            sos = math.sqrt(gamma*prim[i][3]/prim[i][0])
            vel_mag = prim[i][1]**2 + prim[i][2]**2
            mach_number = math.sqrt(vel_mag/sos)
            the_file.write("{} {} {} {} {} {} {} {} {} {} {}\n".format(x[i], y[i], flag_1[i], globaldata[i].qtdepth, prim[i][0], prim[i][1], prim[i][2], prim[i][3], mach_number, entropy[i], sensor[i]))

def computeAdaptSensor(x, y, prim, conn):
    max_sensor = 0
    sensor = np.zeros(len(x), dtype=np.float64)
    D2_dist = sensorD2Distance(x, y, prim, conn)
    for i in range(1, len(x)):
        sensor[i] = D2_dist[i]

    for i in range(1, len(x)):
        if sensor[i] >= max_sensor:
            max_sensor = sensor[i]

    for i in range(1, len(x)):
        sensor[i] = sensor[i] / max_sensor

    return sensor


def sensorD2Distance(x, y, prim, conn):
    u1_i, u2_i, pr_i, rho_i = 0,0,0,0
    u1_j, u2_j, pr_j, rho_j = 0,0,0,0

    maxi, u_sqr = 0,0
    temp1, temp2, temp3 = 0,0,0
    D2_dist = 0
    D2_dist_array = np.zeros(len(x), dtype=np.float64)

    for i in range(1, len(x)):
        u1_i = prim[i][1]
        u2_i = prim[i][2]
        rho_i = prim[i][0]
        pr_i = prim[i][3]

        maxi = 0.0
        for j in conn[i]:
            if j == 0:
                break
            u1_j = prim[j][1]
            u2_j = prim[j][2]
            rho_j = prim[j][0]
            pr_j = prim[j][3]

            temp1 = (pr_i*pr_i*rho_j)/(pr_j*pr_j*rho_i)
            temp1 = math.log(temp1)
            temp1 = (rho_j - rho_i)*temp1

            u_sqr = (u1_j - u1_i)**2 + (u2_j - u2_i)**2
            temp2 = (rho_j/rho_i) + (pr_i*rho_j)/(pr_j*rho_i)
            temp2 = temp2*u_sqr
            temp2 = (0.5*rho_i*rho_i/pr_i)*temp2

            temp3 = rho_i*(((pr_i*rho_j)/(pr_j*rho_i)) - 1)
            temp3 = temp3 + rho_j*(((pr_j*rho_i)/(pr_i*rho_j)) - 1) 
            temp3 = 2*temp3

            D2_dist = temp1 + temp2 + temp3 

            if(D2_dist > maxi):
                maxi = D2_dist

        D2_dist_array[i] = maxi

    return D2_dist_array

def computeEntropy(prim, pr_inf, gamma):
        temp1, temp2 = 0, 0
        total_entropy = 0
        temp2 = math.log(pr_inf)
        entropy = np.zeros(len(prim), dtype=np.float64)
        for i in range(1, len(prim)):
            temp1 = prim[i][0] ** gamma
            temp1 = prim[i][0]**gamma
            temp1 = prim[i][3]/temp1
            temp1 = math.log(temp1)
            entropy[i] = abs(temp1 - temp2)
            total_entropy = total_entropy + abs(temp1 - temp2)
        return entropy