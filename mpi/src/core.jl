function set_local_to_ghost_ID(globaldata, foreignGhostPoints, localPoints, ghostPoints, rank, size, comm)
    for numParts in 0:size-1
        if numParts == rank
            continue
        end

        # To be sent to specific partition with all ghost points info related to that array
        transferArray = zeros(Int, 0) 
        for idx in localPoints + 1:localPoints + ghostPoints
            gPoint = globaldata[idx]
            # If partition number match
            if gPoint.left == numParts
                # Append the localID of that point in that specific parition  
                append!(transferArray, gPoint.right) 
            end
        end
        # Send non-blocking calls. Example (0 => (data, 1,0, comm))
        MPI.Isend(transferArray, numParts, rank, comm)
    end

    requests = Array{MPI.Request,1}(undef, size-1)

    for numParts in 0:size-1
        if numParts == rank
            continue
        end

        # Send non-blocking receives. Example (1 => (buffer, 0, 0, comm))
        receiveSlice = @views foreignGhostPoints[:, numParts+1]
        if numParts < rank
            requests[numParts+1] = MPI.Irecv!(receiveSlice, numParts, numParts, comm)
        else
            requests[numParts] = MPI.Irecv!(receiveSlice, numParts, numParts, comm)
        end
    end
    status = MPI.Waitall!(requests)
end

function getInitialPrimitive(configData)
    rho_inf::Float64 = configData["core"]["rho_inf"]
    mach::Float64 = configData["core"]["mach"]
    machcos::Float64 = mach * cos(calculateTheta(configData))
    machsin::Float64 = mach * sin(calculateTheta(configData))
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
        da = split(itm)
        da1 = parse.(Float64, da)
        push!(finaldata, da1)
    end
    close(dataman)
    return finaldata
end

function placeNormals(globaldata, idx, configData, interior, wall, outer)
    flag = globaldata[idx].flag_1
    if flag == wall || flag == outer
        currpt = getxy(globaldata[idx])
        leftpt = globaldata[idx].left
        leftpt = getxy(globaldata[leftpt])
        rightpt = globaldata[idx].right
        rightpt = getxy(globaldata[rightpt])
        normals = calculateNormals(leftpt, rightpt, currpt[1], currpt[2])
        setNormals(globaldata, idx, normals)
    elseif flag == interior
        setNormals(globaldata, idx, (0,1))
    else
        @warn "Illegal Point Type"
    end
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

    tx = ny
    ty = -nx
    xpos_nbhs = 0
    xneg_nbhs = 0
    ypos_nbhs = 0
    yneg_nbhs = 0
    xpos_conn = SVector{20}([zero(Float64) for iter in 1:20])
    xneg_conn = SVector{20}([zero(Float64) for iter in 1:20])
    yneg_conn = SVector{20}([zero(Float64) for iter in 1:20])
    ypos_conn = SVector{20}([zero(Float64) for iter in 1:20])

    for itm in ptInterest.conn
        if itm == zero(Float64)
            break
        end
        itmx = globaldata[itm].x
        itmy = globaldata[itm].y

        Δx = itmx - currx
        Δy = itmy - curry

        Δs = Δx*tx + Δy*ty
        Δn = Δx*nx + Δy*ny
        if Δs <= 0.0
            xpos_nbhs += 1
            xpos_conn = setindex(xpos_conn, itm, xpos_nbhs)
        end
        if Δs >= 0.0
            xneg_nbhs += 1
            xneg_conn = setindex(xneg_conn, itm, xneg_nbhs)
        end
        if flag == 1
            if Δn <= 0.0
                ypos_nbhs += 1
                ypos_conn = setindex(ypos_conn, itm, ypos_nbhs)
            end
            if Δn >= 0.0
                yneg_nbhs += 1
                yneg_conn = setindex(yneg_conn, itm, yneg_nbhs)
            end
        elseif flag == 0
            yneg_nbhs += 1
            yneg_conn = setindex(yneg_conn, itm, yneg_nbhs)
        elseif flag == 2
            ypos_nbhs += 1
            ypos_conn = setindex(ypos_conn, itm, ypos_nbhs)
        end
    end
    globaldata[idx] = setproperties(globaldata[idx], 
                                    xpos_conn = xpos_conn,
                                    xneg_conn = xneg_conn,
                                    yneg_conn = yneg_conn,
                                    ypos_conn = ypos_conn, 
                                    xpos_nbhs = xpos_nbhs, 
                                    xneg_nbhs = xneg_nbhs, 
                                    ypos_nbhs = ypos_nbhs, 
                                    yneg_nbhs = yneg_nbhs)
    return nothing
end

function getPointDetails(globaldata, point_index)
    println("")
    println(IOContext(stdout, :compact => false), "Q is", globaldata[point_index].q)
    # println(IOContext(stdout, :compact => false), "DQ is", globaldata[point_index].dq)
    println(IOContext(stdout, :compact => false), "Prim is", globaldata[point_index].prim)
    println(IOContext(stdout, :compact => false), "Flux Res is", globaldata[point_index].flux_res)
    # println(IOContext(stdout, :compact => false), "MaxQ is", globaldata[point_index].max_q)
    # println(IOContext(stdout, :compact => false), "MinQ is", globaldata[point_index].min_q)
    println(IOContext(stdout, :compact => false), "Prim Old is", globaldata[point_index].prim_old)
    # println(IOContext(stdout, :compact => false), "Delta is", globaldata[point_index].delta)
end

function fpi_solver(iter, globaldata, configData, res_old, localPoints, ghostPoints, main_store, foreignGhostPoints,
    tempdq, requests_1, requests_2)

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)
    if iter == 1 && rank == 0
        println("Starting FuncDelta for Rank:$rank")
    end

    power = main_store[53]
    cfl = main_store[54]

    @timeit to "func_delta" begin
        func_delta(globaldata, localPoints, cfl)
    end

    phi_i = @view main_store[1:4]
	phi_k = @view main_store[5:8]
	G_i = @view main_store[9:12]
    G_k = @view main_store[13:16]
	result = @view main_store[17:20]
	qtilde_i = @view main_store[21:24]
	qtilde_k = @view main_store[25:28]
	Gxp = @view main_store[29:32]
	Gxn = @view main_store[33:36]
	Gyp = @view main_store[37:40]
	Gyn = @view main_store[41:44]
    ∑_Δx_Δf = @view main_store[45:48]
    ∑_Δy_Δf = @view main_store[49:52]

    if rank == 0
        @printf("Iteration Number %d ", iter)
    end

    for rk in 1:4
        @timeit to "q_var" begin
            q_variables(globaldata, tempdq, localPoints, ghostPoints, result)
        end

        # update_ghost_q(globaldata, foreignGhostPoints, localPoints, ghostPoints, rank, size, comm, tempdq, requests_1)

        # temporarray1 = zeros(Float64,4,4)
        # requests = Array{MPI.Request,1}(undef, 1)
        # if rank == 0
        #     MPI.Isend(zeros(0), 1, 133, comm)
        #     # MPI.Isend([1.2,2.3,3.3,4.3], 1, 134, comm)
        # elseif rank == 1
        #     array_slice = @views temporarray1[1, :]
        #     requests[1] = MPI.Irecv!(array_slice, 0, 133, comm)
        #     # array_slice = @views temporarray1[5:8]
        #     # requests[2] = MPI.Irecv!(array_slice, 0, 134, comm)
        #     status = MPI.Waitall!(requests)
        #     print(status[1].error, "\n")
        #     print(temporarray1, "\n")
        # end

        @timeit to "q_derv" begin
            q_var_derivatives(globaldata, localPoints, power, ∑_Δx_Δf, ∑_Δy_Δf, qtilde_i, qtilde_k)
        end
        
        update_ghost_dq(globaldata, foreignGhostPoints, localPoints, ghostPoints, rank, size, comm, tempdq, requests_1, requests_2)
        update_ghost_maxmin_q(globaldata, foreignGhostPoints, localPoints, ghostPoints, rank, size, comm, tempdq, requests_1, requests_2)

        @timeit to "q_derv_innerloop" begin
            for inner_iters in 1:3
                q_var_derivatives_innerloop(globaldata, localPoints, power, tempdq, ∑_Δx_Δf, ∑_Δy_Δf, qtilde_i, qtilde_k)
                update_ghost_dq(globaldata, foreignGhostPoints, localPoints, ghostPoints, rank, size, comm, tempdq, requests_1, requests_2)
            end
        end

        @timeit to "flux_res" begin
            cal_flux_residual(globaldata, localPoints, configData, Gxp, Gxn, Gyp, Gyn, phi_i, phi_k, G_i, G_k,
                    result, qtilde_i, qtilde_k, ∑_Δx_Δf, ∑_Δy_Δf, main_store)
        end

        @timeit to "state_update" begin
            state_update(globaldata, localPoints, configData, iter, res_old, rk, ∑_Δx_Δf, ∑_Δy_Δf, main_store)
        end

        update_ghost_prim(globaldata, foreignGhostPoints, localPoints, ghostPoints, rank, size, comm, tempdq, requests_1)

    end
    return nothing
end

function q_variables(globaldata, tempdq, localPoints, ghostPoints, q_result)
    for idx in 1:localPoints + ghostPoints
        rho = globaldata.prim[idx][1]
        u1 = globaldata.prim[idx][2]
        u2 = globaldata.prim[idx][3]
        pr = globaldata.prim[idx][4]
        beta = 0.5 * (rho / pr)
        two_times_beta = 2.0 * beta
        q_result[1] = log(rho) + log(beta) * 2.5 - (beta * ((u1 * u1) + (u2 * u2)))
        q_result[2] = (two_times_beta * u1)
        q_result[3] = (two_times_beta * u2)
        q_result[4] = -two_times_beta
        globaldata.q[idx] = SVector{4}(q_result)
    end
    return nothing
end

function q_var_derivatives(globaldata, localPoints, power, ∑_Δx_Δq, ∑_Δy_Δq, max_q, min_q)
    for idx in 1:localPoints
        x_i = globaldata.x[idx]
        y_i = globaldata.y[idx]
        ∑_Δx_sqr = zero(Float64)
        ∑_Δy_sqr = zero(Float64)
        ∑_Δx_Δy = zero(Float64)
        fill!(∑_Δx_Δq, zero(Float64))
        fill!(∑_Δy_Δq, zero(Float64))

        @. max_q = globaldata.q[idx]
        @. min_q = globaldata.q[idx]

        for conn in globaldata.conn[idx]
            if conn == zero(Float64)
                break
            end
            x_k = globaldata.x[conn]
            y_k = globaldata.y[conn]
            Δx = x_k - x_i
            Δy = y_k - y_i
            dist = hypot(Δx, Δy)
            weights = dist ^ power
            ∑_Δx_sqr += (Δx * Δx) * weights
            ∑_Δy_sqr += (Δy * Δy) * weights
            ∑_Δx_Δy += (Δx * Δy) * weights

            for iter in 1:4
                intermediate_var = weights * (globaldata.q[conn][iter] - globaldata.q[idx][iter])
                ∑_Δx_Δq[iter] += Δx * intermediate_var
                ∑_Δy_Δq[iter] += Δy * intermediate_var
            end

            for i in 1:4
                if max_q[i] < globaldata.q[conn][i]
                    max_q[i] = globaldata.q[conn][i]
                end
                if min_q[i] > globaldata.q[conn][i]
                    min_q[i] = globaldata.q[conn][i]
                end
            end
        end
        globaldata.max_q[idx] = SVector{4}(max_q)
        globaldata.min_q[idx] = SVector{4}(min_q)
        q_var_derivatives_update(∑_Δx_sqr, ∑_Δy_sqr, ∑_Δx_Δy, ∑_Δx_Δq, ∑_Δy_Δq, max_q, min_q)
        globaldata.dq1[idx] = SVector{4}(max_q)
        globaldata.dq2[idx] = SVector{4}(min_q)
    end
    return nothing
end

@inline function q_var_derivatives_update(∑_Δx_sqr, ∑_Δy_sqr, ∑_Δx_Δy, ∑_Δx_Δq, ∑_Δy_Δq, dq1_store, dq2_store)
    det = (∑_Δx_sqr * ∑_Δy_sqr) - (∑_Δx_Δy * ∑_Δx_Δy)
    one_by_det = 1.0 / det
    for iter in 1:4
        dq1_store[iter] = one_by_det * (∑_Δx_Δq[iter] * ∑_Δy_sqr - ∑_Δy_Δq[iter] * ∑_Δx_Δy)
        dq2_store[iter] = one_by_det * (∑_Δy_Δq[iter] * ∑_Δx_sqr - ∑_Δx_Δq[iter] * ∑_Δx_Δy)
    end
    return nothing
end

function q_var_derivatives_innerloop(globaldata, localPoints, power, tempdq, ∑_Δx_Δq, ∑_Δy_Δq, qi_tilde, qk_tilde)
    for idx in 1:localPoints
        x_i = globaldata.x[idx]
        y_i = globaldata.y[idx]
        ∑_Δx_sqr = zero(Float64)
        ∑_Δy_sqr = zero(Float64)
        ∑_Δx_Δy = zero(Float64)
        fill!(∑_Δx_Δq, zero(Float64))
        fill!(∑_Δy_Δq, zero(Float64))
        for conn in globaldata.conn[idx]
            if conn == zero(Float64)
                break
            end
            x_k = globaldata.x[conn]
            y_k = globaldata.y[conn]
            Δx = x_k - x_i
            Δy = y_k - y_i
            dist = hypot(Δx, Δy)
            weights = dist ^ power
            ∑_Δx_sqr += (Δx * Δx) * weights
            ∑_Δy_sqr += (Δy * Δy) * weights
            ∑_Δx_Δy += (Δx * Δy) * weights

            q_var_derivatives_get_sum_delq_innerloop(globaldata, idx, conn, weights, Δx, Δy, qi_tilde, qk_tilde, ∑_Δx_Δq, ∑_Δy_Δq)
        end
        det = (∑_Δx_sqr * ∑_Δy_sqr) - (∑_Δx_Δy * ∑_Δx_Δy)
        one_by_det = 1.0 / det
        for iter in 1:4
            tempdq[idx, 1, iter] = one_by_det * (∑_Δx_Δq[iter] * ∑_Δy_sqr - ∑_Δy_Δq[iter] * ∑_Δx_Δy)
            tempdq[idx, 2, iter] = one_by_det * (∑_Δy_Δq[iter] * ∑_Δx_sqr - ∑_Δx_Δq[iter] * ∑_Δx_Δy)
        end 
    end
    for idx in 1:localPoints
        q_var_derivatives_update_innerloop(qi_tilde, qk_tilde, idx, tempdq)
        globaldata.dq1[idx] = SVector{4}(qi_tilde)
        globaldata.dq2[idx] = SVector{4}(qk_tilde)
    end
    return nothing
end

@inline function q_var_derivatives_get_sum_delq_innerloop(globaldata, idx, conn, weights, Δx, Δy, qi_tilde, qk_tilde, ∑_Δx_Δq, ∑_Δy_Δq)
    for iter in 1:4    
        qi_tilde[iter] = globaldata.q[idx][iter] - 0.5 * (Δx * globaldata.dq1[idx][iter] + Δy * globaldata.dq2[idx][iter])
        qk_tilde[iter] = globaldata.q[conn][iter] - 0.5 * (Δx * globaldata.dq1[conn][iter] + Δy * globaldata.dq2[conn][iter])

        intermediate_var = weights * (qk_tilde[iter] - qi_tilde[iter])
        ∑_Δx_Δq[iter] += Δx * intermediate_var
        ∑_Δy_Δq[iter] += Δy * intermediate_var
    end
    return nothing
end

@inline function q_var_derivatives_update_innerloop(dq1, dq2, idx, tempdq)
    for iter in 1:4
        dq1[iter] = tempdq[idx, 1, iter]
        dq2[iter] = tempdq[idx, 2, iter]
    end
    return nothing
end

function update_ghost_q(globaldata, foreignGhostPoints, localPoints, ghostPoints, rank, size, comm, tempdq, requests_1)
    # Send q_vars of foregin ghost points
    for numParts in 0:size-1
        if numParts == rank
            continue
        end

        partitionedForeignGhostPoints = @views foreignGhostPoints[:, numParts+1]
        for iter in eachindex(partitionedForeignGhostPoints)
            indexOfGhostPoint = partitionedForeignGhostPoints[iter]
            if indexOfGhostPoint == 0
                break
            end
            MPI.Isend(MPI.Buffer_send(globaldata.q[indexOfGhostPoint]), numParts, indexOfGhostPoint, comm)
        end
    end

    for idx in localPoints+1:localPoints+ghostPoints
        # if rank == 0 && idx == 12178
        #     println(globaldata.q[idx])
        # end
        originalPartition = globaldata.left[idx]
        originalPointID = globaldata.right[idx]
        q_store = @views tempdq[idx, 1, :]
        requests_1[idx-localPoints] = MPI.Irecv!(q_store, originalPartition, originalPointID, comm)
        # if rank == 0 && idx == 12178
            # println(typeof(q_store))
        #     println(" PartionNo- " ,originalPartition, " PartitionID- ", originalPointID, " Rank- ", rank, " QStore- ", q_store, " Qvalue- ", globaldata.q[idx])
        # end
    end

    status = MPI.Waitall!(requests_1)

    for idx in localPoints+1:localPoints+ghostPoints
        q_store = @views tempdq[idx, 1, :]
        globaldata.q[idx] = q_store
    end
    return nothing
end

function update_ghost_dq(globaldata, foreignGhostPoints, localPoints, ghostPoints, rank, size, comm, tempdq, requests_1, requests_2)
    # Send dq_vars of foregin ghost points
    for numParts in 0:size-1
        if numParts == rank
            continue
        end

        partitionedForeignGhostPoints = @views foreignGhostPoints[:, numParts+1]
        for iter in eachindex(partitionedForeignGhostPoints)
            indexOfGhostPoint = partitionedForeignGhostPoints[iter]
            if indexOfGhostPoint == 0
                break
            end
            # if indexOfGhostPoint == 1 && rank  == 1
            #     print(" The dqs are for Rank:$rank", globaldata.dq1[indexOfGhostPoint], " and ", globaldata.dq2[indexOfGhostPoint],"\n")
            # end
            MPI.Isend(MPI.Buffer_send(globaldata.dq1[indexOfGhostPoint]), numParts, 2*indexOfGhostPoint, comm)
            MPI.Isend(MPI.Buffer_send(globaldata.dq2[indexOfGhostPoint]), numParts, 2*indexOfGhostPoint + 1, comm)
        end
    end

    for idx in localPoints+1:localPoints+ghostPoints
        originalPartition = globaldata.left[idx]
        originalPointID = globaldata.right[idx]
        dq_store = @views tempdq[idx, 1, :]
        requests_1[idx-localPoints] = MPI.Irecv!(dq_store, originalPartition, 2*originalPointID, comm)
        dq_store = @views tempdq[idx, 2, :]
        requests_2[idx-localPoints] = MPI.Irecv!(dq_store, originalPartition, 2*originalPointID + 1, comm)
    end

    status = MPI.Waitall!(requests_1)
    status = MPI.Waitall!(requests_2)

    for idx in localPoints+1:localPoints+ghostPoints
        # if globaldata.right[idx] == 1 && rank == 0
        #     print(" The temps are for Rank:$rank", tempdq[idx, 1, :], " and ", tempdq[idx, 2, :],"\n")
        # end
        dq_store = @views tempdq[idx, 1, :]
        globaldata.dq1[idx] = dq_store
        dq_store = @views tempdq[idx, 2, :]
        globaldata.dq2[idx] = dq_store
        # if globaldata.right[idx] == 1 && rank == 0
        #     print(" The updated dqs are for Rank:$rank", globaldata.dq1[idx], " and ", globaldata.dq2[idx],"\n")
        # end
    end
    return nothing
end

function update_ghost_maxmin_q(globaldata, foreignGhostPoints, localPoints, ghostPoints, rank, size, comm, tempdq, requests_1, requests_2)
    # Send dq_vars of foregin ghost points
    for numParts in 0:size-1
        if numParts == rank
            continue
        end

        partitionedForeignGhostPoints = @views foreignGhostPoints[:, numParts+1]
        for iter in eachindex(partitionedForeignGhostPoints)
            indexOfGhostPoint = partitionedForeignGhostPoints[iter]
            if indexOfGhostPoint == 0
                break
            end
            MPI.Isend(MPI.Buffer_send(globaldata.max_q[indexOfGhostPoint]), numParts, 2*indexOfGhostPoint, comm)
            MPI.Isend(MPI.Buffer_send(globaldata.min_q[indexOfGhostPoint]), numParts, 2*indexOfGhostPoint + 1, comm)
        end
    end

    for idx in localPoints+1:localPoints+ghostPoints
        # if rank == 0 && idx == 12178
        #     println(globaldata.q[idx])
        # end
        originalPartition = globaldata.left[idx]
        originalPointID = globaldata.right[idx]
        dq_store = @views tempdq[idx, 1, :]
        requests_1[idx-localPoints] = MPI.Irecv!(dq_store, originalPartition, 2*originalPointID, comm)
        dq_store = @views tempdq[idx, 2, :]
        requests_2[idx-localPoints] = MPI.Irecv!(dq_store, originalPartition, 2*originalPointID + 1, comm)
    end

    status = MPI.Waitall!(requests_1)
    status = MPI.Waitall!(requests_2)

    for idx in localPoints+1:localPoints+ghostPoints
        dq_store = @views tempdq[idx, 1, :]
        globaldata.max_q[idx] = dq_store
        dq_store = @views tempdq[idx, 2, :]
        globaldata.min_q[idx] = dq_store
    end
    return nothing
end

function update_ghost_prim(globaldata, foreignGhostPoints, localPoints, ghostPoints, rank, size, comm, tempdq, requests_1)
    # Send q_vars of foregin ghost points
    for numParts in 0:size-1
        if numParts == rank
            continue
        end

        partitionedForeignGhostPoints = @views foreignGhostPoints[:, numParts+1]
        for iter in eachindex(partitionedForeignGhostPoints)
            indexOfGhostPoint = partitionedForeignGhostPoints[iter]
            if indexOfGhostPoint == 0
                break
            end
            MPI.Isend(MPI.Buffer_send(globaldata.prim[indexOfGhostPoint]), numParts, indexOfGhostPoint, comm)
        end
    end

    for idx in localPoints+1:localPoints+ghostPoints
        originalPartition = globaldata.left[idx]
        originalPointID = globaldata.right[idx]
        prim_store = @views tempdq[idx, 1, :]
        requests_1[idx-localPoints] = MPI.Irecv!(prim_store, originalPartition, originalPointID, comm)
    end

    status = MPI.Waitall!(requests_1)

    for idx in localPoints+1:localPoints+ghostPoints
        prim_store = @views tempdq[idx, 1, :]
        globaldata.prim[idx] = prim_store
    end
    return nothing
end

    # tempor = globaldata.q[1]
    # tempor = globaldata[1].q
    # tempor = Array(globaldata[1].q)
    # tempor = globaldata.q[1:10]
    # tempor = [Dict('a'=>2, 'b'=>3), Dict('f'=>8, 'r'=>3)]
    # tempor = globaldata[1:3]
    # tempor = [Base.ImmutableDict("a"=>2), Base.ImmutableDict("f"=>8)]
    # tempor = spzeros(3,3)
    # tempor = globaldata.q[1]
    # print(tempor, "\n")
    # println(isbitstype(eltype(globaldata.q[1])))
    # x= @SVector [1, 2, 3,4]
    # MPI.Gather(x, 0, MPI.COMM_WORLD)
    # print(q_result, "\n")
    # MPI.Bcast!(MPI.Buffer_send(tempor), 0, MPI.COMM_WORLD)
