import math

def generateOutput(globaldata):
    computeAdaptSensor(globaldata)

def computeAdaptSensor(globaldata):
    max_sensor = 0
    

def sensorD2Distance(globaldata):
    sensor = ["start"]
    for idx in range(len(globaldata)):
        if idx > 0:
            u1_i = globaldata[idx].prim[1]
            u2_i = globaldata[idx].prim[2]
            rho_i = globaldata[idx].prim[0]
            pr_i = globaldata[idx].prim[3]
            x_i = globaldata[idx].x
            y_i = globaldata[idx].y

            maxi = 0
            mini = 1e10

            for itm in globaldata[idx].conn:
                u1_j = globaldata[itm].prim[1]
                u2_j = globaldata[itm].prim[2]
                rho_j = globaldata[itm].prim[0]
                pr_j = globaldata[itm].prim[3]
                x_j = globaldata[itm].x
                y_j = globaldata[itm].y

                dist_ij = (x_j - x_i)**2 + (y_j - y_i)**2

                temp1 = (pr_i * pr_i * rho_j) / (pr_j * pr_j * rho_i)
                temp1 = math.log(temp1)
                temp1 = (rho_j - rho_i) * temp1

                u_sqr = (u1_j - u1_i) ** 2 + (u2_j - u2_i) ** 2
                temp2 = (rho_j / rho_i) + (pr_i * rho_j) / (pr_j * rho_i)
                temp2 = temp2 * u_sqr
                temp2 = (0.5 * rho_i * rho_i / pr_i) * temp2

                temp3 = rho_i*(((pr_i*rho_j)/(pr_j*rho_i)) - 1)
                temp3 = temp3 + rho_j*(((pr_j*rho_i)/(pr_i*rho_j)) - 1)
                temp3 = 2*temp3

                D2_dist = temp1 + temp2 + temp3

                if (D2_dist > maxi):
                    maxi = D2_dist 
                sensor.append(maxi)
            