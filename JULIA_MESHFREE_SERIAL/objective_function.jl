
# TODO - Check deg2rad
function calculateTheta(configData)
    theta = deg2rad(Float64(configData["core"]["aoa"]))::Float64
    return theta
end

function compute_cl_cd_cm(globaldata, configData, shapeindices)

    rho_inf::Float64 = configData["core"]["rho_inf"]::Float64
    Mach::Float64 = configData["core"]["mach"]::Float64
    pr_inf::Float64 = configData["core"]["pr_inf"]::Float64
    shapes::Int64 = configData["core"]["shapes"]::Float64
    theta = calculateTheta(configData)

    temp = 0.5*rho_inf*Mach*Mach

    H = zeros(Float64, shapes)
    V = zeros(Float64, shapes)
    pitch_mom = zeros(Float64, shapes)

    Cl = zeros(Float64, shapes)
    Cd = zeros(Float64, shapes)
    Cm = zeros(Float64, shapes)

    open("cp_file", "w") do the_file
        for itm in shapeindices
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
            cp = globaldata[itm].prim[4] - pr_inf
            cp = -cp/temp

            flag_2 = globaldata[itm].flag_2

            print(the_file, flag_2, " ", mx, " ",  cp, "\n")
            H[flag_2] += cp * nx * ds
            V[flag_2] += cp * ny * ds

            pitch_mom[flag_2] += (-cp * ny * ds * (mx - 0.25)) + (cp * nx * ds * my)
        end
    end

    Cl = V*cos(theta) - H*sin(theta)
    Cd = H*cos(theta) + V*sin(theta)
    Cm = pitch_mom

end
