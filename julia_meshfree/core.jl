function getInitialPrimitive(configData)
    rho_inf = parse(Float64, configData["core"]["rho_inf"])
    mach = parse(Float64, configData["core"]["mach"])
    machcos = mach * cos(calculateTheta(configData))
    machsin = mach * sin(calculateTheta(configData))
    pr_inf = parse(Float64, configData["core"]["pr_inf"])
    primal = [rho_inf, machcos, machsin, pr_inf]
    return primal
end

function getInitialPrimitive2(configData)
    dataman = open("prim_soln_clean")
    data = read(dataman, String)
    data = split(data, "\n")
    finaldata = []
    for (idx,itm) in enumerate(data)
        # try
        da = split(itm)
        # TODO - Check this list comprehension
        # da = list(map(float, da))
        # print(da)
        da = parse.(Float64, da)
        push!(finaldata, da)
        # catch
        #     print(idx)
        # end
    end
    # print(length(finaldata))
    return finaldata
end

function calculateNormals(left, right, mx, my)
    lx = left[1]
    ly = left[2]

    rx = right[1]
    ry = right[2]

    nx1 = my - ly
    nx2 = ry - my

    ny1 = mx - lx
    ny2 = rx - mx

    nx = 0.5*(nx1 + nx2)
    ny = 0.5*(ny1 + ny2)

    det = sqrt(nx*nx + ny*ny)

    nx = -nx/det
    ny = ny/det

    return (nx,ny)
end

function calculateConnectivity(globaldata, idx)
    ptInterest = globaldata[idx]
    currx = ptInterest.x
    curry = ptInterest.y
    nx = ptInterest.nx
    ny = ptInterest.ny

    flag = ptInterest.flag_1

    xpos_conn,xneg_conn,ypos_conn,yneg_conn = [],[],[],[]

    tx = ny
    ty = -nx

    # if idx == 1
    #     println("\n\n ")
    #     println(ptInterest)
    #     println(currx)
    #     println(curry)
    #     println(nx)
    #     println(ny)
    #     println(flag)
    #     println(tx)
    #     println(ty)
    # end
    for itm in ptInterest.conn
        itmx = globaldata[itm].x
        itmy = globaldata[itm].y

        delx = itmx - currx
        dely = itmy - curry

        dels = delx*tx + dely*ty
        deln = delx*nx + dely*ny

        if dels <= 0
            push!(xpos_conn, itm)
        end
        if dels >= 0
            push!(xneg_conn, itm)
        end
        if flag == 2
            if deln <= 0
                push!(ypos_conn, itm)
            end
            if deln >= 0
                push!(yneg_conn, itm)
            end
        elseif flag == 1
            push!(yneg_conn, itm)
        elseif flag == 3
            push!(ypos_conn, itm)
        end
    end
    return (xpos_conn, xneg_conn, ypos_conn, yneg_conn)
end

function fpi_solver(iter, globaldata, configData, wallindices, outerindices, interiorindices, res_old)
    # print(globaldata[77].prim)
    # print(" 111\n")
    globaldata = q_var_derivatives(globaldata, configData)
    # print(globaldata[77].prim)
    # print(" 222\n")
    globaldata = cal_flux_residual(globaldata, wallindices, outerindices, interiorindices, configData)
    # print(globaldata[77].prim)
    # print(" 333\n")
    globaldata = func_delta(globaldata, configData)
    # print(globaldata[77].prim)
    # print(" 444\n")
    globaldata, res_old = state_update(globaldata, wallindices, outerindices, interiorindices, configData, iter, res_old)
    # print(globaldata[77].prim)
    # print(" 555\n")
    compute_cl_cd_cm(globaldata, configData, wallindices)
    # print(globaldata[77].prim)
    # print(" 666\n")
    return res_old, globaldata
end

function q_var_derivatives(globaldata, configData)
    power = Int(configData["core"]["power"])

    for (idx, itm) in enumerate(globaldata)
        if idx > 0
            rho = 1
            tempq = zeros(Float64, 4)
            rho = itm.prim[1]
            u1 = itm.prim[2]
            u2 = itm.prim[3]
            pr = itm.prim[4]

            # print(rho)
            # print("\n")

            beta = 0.5 * (itm.prim[1] / itm.prim[4])
            try
                a = log(rho)
                b = log(beta)
            catch
                println(rho)
                println(beta)
                println(idx)
            end

            tempq[1] = log(rho) + log(beta) * 2.5 - (beta * ((u1 * u1) + (u2 * u2)))
            two_times_beta = 2 * beta

            tempq[2] = (two_times_beta * u1)
            tempq[3] = (two_times_beta * u2)
            tempq[4] = -two_times_beta

            globaldata[idx].q = tempq
        end
    end

    for (idx,itm) in enumerate(globaldata)
        if idx > 0

            x_i = itm.x
            y_i = itm.y

            sum_delx_sqr = 0
            sum_dely_sqr = 0
            sum_delx_dely = 0

            sum_delx_delq = zeros(Float64, 4)
            sum_dely_delq = zeros(Float64, 4)

            for conn in itm.conn
                # print(conn)
                # print(" 123 \n")
                # print(itm.conn)
                x_k = globaldata[conn].x
                # print(x_k)
                # print("   \n\n")
                y_k = globaldata[conn].y

                delx = x_k - x_i
                dely = y_k - y_i

                dist = sqrt(delx*delx + dely*dely)
                weights = dist ^ power

                sum_delx_sqr = sum_delx_sqr + ((delx * delx) * weights)
                sum_dely_sqr = sum_dely_sqr + ((dely * dely) * weights)

                sum_delx_dely = sum_delx_dely + ((delx * dely) * weights)

                sum_delx_delq = sum_delx_delq + (weights * delx * (globaldata[conn].q - globaldata[idx].q))

                sum_dely_delq = sum_dely_delq + (weights * dely * (globaldata[conn].q - globaldata[idx].q))
            end

            det = (sum_delx_sqr * sum_dely_sqr) - (sum_delx_dely * sum_delx_dely)
            one_by_det = 1 / det

            tempdq = zeros(Float64, 2)

            sum_delx_delq1 = sum_delx_delq * sum_dely_sqr
            sum_dely_delq1 = sum_dely_delq * sum_delx_dely

            tempsumx = one_by_det * (sum_delx_delq1 - sum_dely_delq1)

            sum_dely_delq2 = sum_dely_delq * sum_delx_sqr

            sum_delx_delq2 = sum_delx_delq * sum_delx_dely

            tempsumy = one_by_det * (sum_dely_delq2 - sum_delx_delq2)

            tempdq = [tempsumx, tempsumy]

            globaldata[idx].dq = tempdq
        end
    end
    return globaldata
end

# function q_var_derivatives_cuda(globaldata, config)
#     tpl = Template("""
#             struct Point{
#                 int localID;
#                 double x;
#                 double y;
#                 int left;
#                 int right;
#                 int flag_1;
#                 int flag_2;
#                 int* nbhs;
#                 int* conn;
#                 float nx;
#                 float ny;
#                 float* prim;
#                 float* flux_res;
#                 float* q;
#                 float* dq;
#                 float* entropy;
#                 int xpos_nbhs;
#                 int xneg_nbhs;
#                 int ypos_nbhs;
#                 int yneg_nbhs;
#                 int* xpos_conn;
#                 int* xneg_conn;
#                 int* ypos_conn;
#                 int* yneg_conn;
#                 float delta;
#             };

#             __global__ void q_var_derivatives()
#             {

#                 int i = (blockIdx.x)* blockDim.x + threadIdx.x;
#                 int j = (blockIdx.y)* blockDim.y + threadIdx.y;

#                 int width = {{ POINT_WIDTH }};

#                 double r = {{ DELTA }};

#                 double u1 = u_old[i + (width * j)];
#                 double ul = u_old[(i-1) + (width * j)];
#                 double ur = u_old[(i+1) + (width * j)];
#                 double utop = u_old[i + (width * (j+1))];
#                 double ubottom = u_old[i + (width * (j-1))];
#                 double test = 0;

#                 if (i > 0 && i < {{ POINT_SIZE_X }} - 1 && j > 0 && j < {{ POINT_SIZE_Y }} - 1){
#                     test = u1 + (r * (ul + ur + utop + ubottom - (4 * u1)));
#                     u_new[i + width * j] = test;
#                 }


#             }""")

#     rendered_tpl = tpl.render(POINT_SIZE_X=1, POINT_SIZE_Y=2, POINT_WIDTH=3, DELTA=4)
#     mod = SourceModule(rendered_tpl)
#     heatCalculate = mod.get_function("heatCalculate")
# end
