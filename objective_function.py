import math
import core
import numpy as np

def compute_cl_cd_cm(globaldata, configData, wallindices, comm=None):

    rho_inf = configData["core"]["rho_inf"]
    Mach = configData["core"]["mach"]
    pr_inf = configData["core"]["pr_inf"]
    shapes = configData["core"]["shapes"]
    theta = core.calculateTheta(configData)

    temp = 0.5*rho_inf*Mach*Mach

    H = [0 for i in range(shapes)]
    V = [0 for i in range(shapes)]
    pitch_mom = [0 for i in range(shapes)]

    Cl = [0 for i in range(shapes)]
    Cd = [0 for i in range(shapes)]
    Cm = [0 for i in range(shapes)]

    if comm != None:
        file_id = "_" + str(comm.Get_rank())
    else:
        file_id = ""

    with open('cp_file%s' % file_id, 'a') as the_file:

        for itm in wallindices:
            left = globaldata[itm].left
            right = globaldata[itm].right
            lx = globaldata[left].x
            ly = globaldata[left].y
            rx = globaldata[right].x
            ry = globaldata[right].y
            mx = globaldata[itm].x
            my = globaldata[itm].y

            ds1 = (mx - lx)**2 + (my - ly)**2
            ds1 = math.sqrt(ds1)

            ds2 = (rx - mx)**2 + (ry - my)**2
            ds2 = math.sqrt(ds2)

            ds = 0.5*(ds1 + ds2)


            nx = globaldata[itm].nx
            ny = globaldata[itm].ny

            cp = globaldata[itm].prim[2] - pr_inf
            cp = -cp/temp

            flag_2 = globaldata[itm].flag_2 - 1

            the_file.write(("%i %f %f\n") % (flag_2, mx, cp))

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