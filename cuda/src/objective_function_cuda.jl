
# TODO - Check deg2rad
function calculateTheta(configData)
    theta = deg2rad(Float64(configData["core"]["aoa"]))
    return theta
end

function compute_cl_cd_cm(globalDataFixedPoint, globalDataPrim, configData)

    rho_inf::Float64 = configData["core"]["rho_inf"]
    Mach::Float64 = configData["core"]["mach"]
    pr_inf::Float64 = configData["core"]["pr_inf"]
    shapes::Int64 = configData["core"]["shapes"]
    wall = configData["point"]["wall"]
    theta = calculateTheta(configData)

    temp = 0.5*rho_inf*Mach*Mach

    H = zeros(Float64, shapes)
    V = zeros(Float64, shapes)
    pitch_mom = zeros(Float64, shapes)

    Cl = zeros(Float64, shapes)
    Cd = zeros(Float64, shapes)
    Cm = zeros(Float64, shapes)
    println("===CP File===")
    open("cp_file_cuda.txt", "w") do the_file
        for (idx, itm) in enumerate(globalDataFixedPoint)
            # print(wallindices)
            if globalDataFixedPoint[idx].flag_1 == wall
                left = globalDataFixedPoint[idx].left
                right = globalDataFixedPoint[idx].right
                lx = globalDataFixedPoint[left].x
                ly = globalDataFixedPoint[left].y
                rx = globalDataFixedPoint[right].x
                ry = globalDataFixedPoint[right].y
                mx = globalDataFixedPoint[idx].x
                my = globalDataFixedPoint[idx].y

                ds1 = hypot(mx - lx, my - ly)
                ds2 = hypot(rx - mx, ry - my)

                ds = 0.5*(ds1 + ds2)
                nx = globalDataFixedPoint[idx].nx
                ny = globalDataFixedPoint[idx].ny
                cp = globalDataPrim[idx, 4] - pr_inf
                cp = -cp/temp

                flag_2 = globalDataFixedPoint[idx].flag_2

                print(the_file, flag_2, " ", mx, " ",  cp, "\n")
                H[flag_2] += cp * nx * ds
                V[flag_2] += cp * ny * ds

                pitch_mom[flag_2] += (-cp * ny * ds * (mx - 0.25)) + (cp * nx * ds * my)
            end
        end
    end

    Cl = V*cos(theta) - H*sin(theta)
    Cd = H*cos(theta) + V*sin(theta)
    Cm = pitch_mom
end
