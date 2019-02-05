
# TODO - Check deg2rad
function calculateTheta(configData)
    theta = deg2rad(Float64(configData["core"]["aoa"]))
    return theta
end

function compute_cl_cd_cm(globaldata, configData, wallindices)

    rho_inf = configData["core"]["rho_inf"]
    Mach = configData["core"]["mach"]
    pr_inf = configData["core"]["pr_inf"]
    shapes = configData["core"]["shapes"]
    theta = calculateTheta(configData)

    temp = 0.5*rho_inf*Mach*Mach

    H = [0.0 for i in 1:shapes]
    V = [0.0 for i in 1:shapes]
    pitch_mom = [0.0 for i in 1:shapes]

    Cl = [0.0 for i in 1:shapes]
    Cd = [0.0 for i in 1:shapes]
    Cm = [0.0 for i in 1:shapes]

    open("cp_file", "a") do the_file
        for itm in wallindices
            # print(wallindices)
            left = globaldata[itm].left
            right = globaldata[itm].right
            lx = globaldata[left].x
            ly = globaldata[left].y
            rx = globaldata[right].x
            ry = globaldata[right].y
            mx = globaldata[itm].x
            my = globaldata[itm].y

            ds1 = hypot(mx - lx, my - ly)
            ds2 = hypot(rx - mx, ry - my)

            ds = 0.5*(ds1 + ds2)


            nx = globaldata[itm].nx
            ny = globaldata[itm].ny

            cp = globaldata[itm].prim[3] - pr_inf
            cp = -cp/temp

            flag_2 = globaldata[itm].flag_2

            write(the_file, flag_2, " ", mx, " ",  cp, "\n")
            H[flag_2] += cp * nx * ds
            V[flag_2] += cp * ny * ds

            pitch_mom[flag_2] += (-cp * ny * ds * (mx - 0.25)) + (cp * nx * ds * my)
        end
    end

    # V = np.array(V)
    # H = np.array(H)

    # TODO - Make this work
    Cl = V*cos(theta) - H*sin(theta)
    Cd = H*cos(theta) + V*sin(theta)
    Cm = pitch_mom

    # if configData["core"]["clcd_flag"]
    #     print("Cl",Cl)
    #     print("Cd",Cd)
    # end
end
