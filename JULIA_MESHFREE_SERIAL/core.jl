function getInitialPrimitive(configData)
    rho_inf = configData["core"]["rho_inf"]::Float64
    mach = configData["core"]["mach"]::Float64
    machcos::Float64 = mach * cos(calculateTheta(configData))
    machsin::Float64 = mach * sin(calculateTheta(configData))
    pr_inf = configData["core"]["pr_inf"]::Float64
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

    xpos_conn,xneg_conn,ypos_conn,yneg_conn = Array{Int32,1}(undef, 0),Array{Int32,1}(undef, 0),Array{Int32,1}(undef, 0),Array{Int32,1}(undef, 0)

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

function fpi_solver(iter, globaldata, configData, wallindices::Array{Int32,1}, outerindices::Array{Int32,1}, interiorindices::Array{Int32,1}, res_old)
    # println(IOContext(stdout, :compact => false), globaldata[3].prim)
    # print(" 111\n")
    if iter == 1
        println("Starting QVar")
    end
    q_var_derivatives(globaldata, configData)
    # println(IOContext(stdout, :compact => false), globaldata[3].prim)
    if iter == 1
        println("Starting Calflux")
    end
    cal_flux_residual(globaldata, wallindices, outerindices, interiorindices, configData)
    # println(IOContext(stdout, :compact => false), globaldata[3].prim)
    if iter == 1
        println("Starting FuncDelta")
    end
    func_delta(globaldata, configData)
    # println(IOContext(stdout, :compact => false), globaldata[3].prim)
    # residue = 0
    if iter == 1
        println("Starting StateUpdate")
    end
    state_update(globaldata, wallindices, outerindices, interiorindices, configData, iter, res_old)
    # println(IOContext(stdout, :compact => false), globaldata[3].prim)
    # residue = res_old
    return nothing
end

function q_var_derivatives(globaldata::Array{Point,1}, configData)
    power::Float64 = configData["core"]["power"]::Float64

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
    # println(IOContext(stdout, :compact => false), globaldata[3].q)
    sum_delx_delq = zeros(Float64, 4)
    sum_dely_delq = zeros(Float64, 4)
    for (idx,itm) in enumerate(globaldata)
        x_i = itm.x
        y_i = itm.y
        sum_delx_sqr = zero(Float64)
        sum_dely_sqr = zero(Float64)
        sum_delx_dely = zero(Float64)
        sum_delx_delq = fill!(sum_delx_delq, 0.0)
        sum_dely_delq = fill!(sum_dely_delq, 0.0)
        for i in 1:4
            globaldata[idx].max_q[i] = globaldata[idx].q[i]
            globaldata[idx].min_q[i] = globaldata[idx].q[i]
        end
        for conn in itm.conn
            x_k = globaldata[conn].x
            y_k = globaldata[conn].y
            delx = x_k - x_i
            dely = y_k - y_i
            dist = hypot(delx, dely)
            weights = dist ^ power
            sum_delx_sqr += ((delx * delx) * weights)
            sum_dely_sqr += ((dely * dely) * weights)
            sum_delx_dely += ((delx * dely) * weights)
            sum_delx_delq += @. (weights * delx * (globaldata[conn].q - globaldata[idx].q))
            sum_dely_delq += @. (weights * dely * (globaldata[conn].q - globaldata[idx].q))
            for i in 1:4
                if globaldata[idx].max_q[i] < globaldata[conn].q[i]
                    globaldata[idx].max_q[i] = globaldata[conn].q[i]
                end
                if globaldata[idx].min_q[i] > globaldata[conn].q[i]
                    globaldata[idx].min_q[i] = globaldata[conn].q[i]
                end
            end
        end
        det = (sum_delx_sqr * sum_dely_sqr) - (sum_delx_dely * sum_delx_dely)
        one_by_det = 1.0 / det
        globaldata[idx].dq[1] = @. one_by_det * (sum_delx_delq * sum_dely_sqr - sum_dely_delq * sum_delx_dely)
        globaldata[idx].dq[2] = @. one_by_det * (sum_dely_delq * sum_delx_sqr - sum_delx_delq * sum_delx_dely)
        # globaldata[idx].dq = [tempsumx, tempsumy]
    end
    # println(IOContext(stdout, :compact => false), globaldata[3].dq)
    # println(IOContext(stdout, :compact => false), globaldata[3].max_q)
    # println(IOContext(stdout, :compact => false), globaldata[3].min_q)
    return nothing
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
