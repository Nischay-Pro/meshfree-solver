function getInitialPrimitive(configData)
    rho_inf::Float64 = configData["core"]["rho_inf"]
    mach::Float64 = configData["core"]["mach"]
    machcos = mach * cos(calculateTheta(configData))
    machsin = mach * sin(calculateTheta(configData))
    pr_inf::Float64 = configData["core"]["pr_inf"]
    primal = [rho_inf, machcos, machsin, pr_inf]
    return primal
end

function getInitialPrimitive2(configData)
    dataman = open("prim_soln_clean")
    data = read(dataman, String)
    data1 = split(data, "\n")
    finaldata = Array{Array{Float64,1},1}(undef, 0)
    for (idx,itm) in enumerate(data1)
        # try
        da = split(itm)
        da1 = parse.(Float64, da)
        push!(finaldata, da1)
    end
    close(dataman)
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

    det = hypot(nx, ny)

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

    xpos_conn,xneg_conn,ypos_conn,yneg_conn = Array{Int,1}(undef, 0),Array{Int,1}(undef, 0),Array{Int,1}(undef, 0),Array{Int,1}(undef, 0)

    tx = ny
    ty = -nx

    for itm in ptInterest.conn
        itmx = globaldata[itm].x
        itmy = globaldata[itm].y

        delx = itmx - currx
        dely = itmy - curry

        dels = delx*tx + dely*ty
        deln = delx*nx + dely*ny
        if dels <= 0.0
            push!(xpos_conn, itm)
        end
        if dels >= 0.0
            push!(xneg_conn, itm)
        end
        if flag == 2
            if deln <= 0.0
                push!(ypos_conn, itm)
            end
            if deln >= 0.0
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
    # println(IOContext(stdout, :compact => false), globaldata[1].prim)
    # print(" 111\n")

    q_var_derivatives(globaldata, configData)
    println("q1")
    println(IOContext(stdout, :compact => false), globaldata[1].q)
    # println("Prim")
    # println(IOContext(stdout, :compact => false), globaldata[2].prim)
    # print(" 222\n")
    # println(IOContext(stdout, :compact => false), globaldata[100].dq)
    cal_flux_residual(globaldata, wallindices, outerindices, interiorindices, configData)
    # println(IOContext(stdout, :compact => false), globaldata[1].prim)
    # print(" 333\n")
    func_delta(globaldata, configData)
    # if iter > 49
    #     println(IOContext(stdout, :compact => false), globaldata[100].flux_res)
    # end
    # println(IOContext(stdout, :compact => false), globaldata[100].delta)
    # println(IOContext(stdout, :compact => false), globaldata[1].prim)
    # print(" 444\n")
    # println("Prim1")
    # println(IOContext(stdout, :compact => false), globaldata[1].prim)
    state_update(globaldata, wallindices, outerindices, interiorindices, configData, iter, res_old)
    # if iter > 49
    #     println(IOContext(stdout, :compact => false), globaldata[1].prim)
    # end
    # println(IOContext(stdout, :compact => false), globaldata[1].prim)
    # print(" 555\n")
    # compute_cl_cd_cm(globaldata, configData, wallindices)
    # println("Prim2")
    # println(IOContext(stdout, :compact => false), globaldata[1].prim)
    # print(" 666\n")
    println("q2")
    println(IOContext(stdout, :compact => false), globaldata[1].q)
end

function q_var_derivatives(globaldata, configData)
    power::Float64 = configData["core"]["power"]

    for (idx, itm) in enumerate(globaldata)
        rho = itm.prim[1]
        u1 = itm.prim[2]
        u2 = itm.prim[3]
        pr = itm.prim[4]

        beta::Float64 = 0.5 * (rho / pr)
        globaldata[idx].q[1] = log(rho) + log(beta) * 2.5 - (beta * ((u1 * u1) + (u2 * u2)))
        two_times_beta = 2.0 * beta
        # if idx == 1
        #     println(globaldata[idx].q[1])
        # end
        globaldata[idx].q[2] = (two_times_beta * u1)
        globaldata[idx].q[3] = (two_times_beta * u2)
        globaldata[idx].q[4] = -two_times_beta
    end

    for (idx,itm) in enumerate(globaldata)
        x_i = itm.x
        y_i = itm.y
        sum_delx_sqr = zero(Float64)
        sum_dely_sqr = zero(Float64)
        sum_delx_dely = zero(Float64)
        sum_delx_delq = zeros(Float64, 4)
        sum_dely_delq = zeros(Float64, 4)
        for conn in itm.conn
            x_k = globaldata[conn].x
            y_k = globaldata[conn].y
            delx = x_k - x_i
            dely = y_k - y_i
            dist = hypot(delx, dely)
            weights = dist ^ power
            sum_delx_sqr +=(delx * delx) * weights
            sum_dely_sqr += (dely * dely) * weights
            sum_delx_dely += (delx * dely) * weights
            sum_delx_delq += weights * delx * (globaldata[conn].q - globaldata[idx].q)
            sum_dely_delq += weights * dely * (globaldata[conn].q - globaldata[idx].q)
        end
        det = (sum_delx_sqr * sum_dely_sqr) - (sum_delx_dely * sum_delx_dely)
        one_by_det = 1 / det
        sum_delx_delq1 = sum_delx_delq * sum_dely_sqr
        sum_dely_delq1 = sum_dely_delq * sum_delx_dely
        tempsumx = one_by_det * (sum_delx_delq1 - sum_dely_delq1)
        sum_dely_delq2 = sum_dely_delq * sum_delx_sqr
        sum_delx_delq2 = sum_delx_delq * sum_delx_dely
        tempsumy = one_by_det * (sum_dely_delq2 - sum_delx_delq2)
        globaldata[idx].dq = [tempsumx, tempsumy]
    end
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
