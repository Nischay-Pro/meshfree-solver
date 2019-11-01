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