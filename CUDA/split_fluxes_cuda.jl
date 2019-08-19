function flux_Gxp_kernel(nx, ny, idx, shared, op::Function, thread_idx, block_dim)

    u1 = shared[thread_idx + block_dim * 4]
    u2 = shared[thread_idx + block_dim * 5]
    rho = shared[thread_idx + block_dim * 6]
    pr = shared[thread_idx + block_dim * 7]

    ut = u1*ny - u2*nx
    un = u1*nx + u2*ny

    beta = 0.5*rho/pr
    S1 = ut*CUDAnative.sqrt(beta)
    B1 = 0.5*CUDAnative.exp(-S1*S1)/CUDAnative.sqrt(pi*beta)
    A1pos = 0.5*(1 + CUDAnative.erf(S1))

    pr_by_rho = pr/rho
    u_sqr = ut*ut + un*un

    shared[thread_idx] = op((rho*(ut*A1pos + B1)), shared[thread_idx])
    temp1 = pr_by_rho + ut*ut
    temp2 = temp1*A1pos + ut*B1
    shared[thread_idx + block_dim] = op((rho*temp2), shared[thread_idx + block_dim])
    temp1 = ut*un*A1pos + un*B1
    shared[thread_idx + block_dim * 2] = op((rho*temp1), shared[thread_idx + block_dim * 2])

    temp1 = (7*pr_by_rho) + u_sqr
    temp2 = 0.5*ut*temp1*A1pos
    temp1 = (6*pr_by_rho) + u_sqr
    shared[thread_idx + block_dim * 3] = op((rho*(temp2 + 0.5*temp1*B1)), shared[thread_idx + block_dim * 3])

    return nothing
end

function flux_Gxn_kernel(nx, ny, idx, shared, op::Function, thread_idx, block_dim)

    u1 = shared[thread_idx + block_dim * 4]
    u2 = shared[thread_idx + block_dim * 5]
    rho = shared[thread_idx + block_dim * 6]
    pr = shared[thread_idx + block_dim * 7]

    ut = u1*ny - u2*nx
    un = u1*nx + u2*ny

    beta = 0.5*rho/pr
    S1 = ut*CUDAnative.sqrt(beta)
    B1 = 0.5*CUDAnative.exp(-S1*S1)/CUDAnative.sqrt(pi*beta)
    A1neg = 0.5*(1 - CUDAnative.erf(S1))

    pr_by_rho = pr/rho
    u_sqr = ut*ut + un*un
    shared[thread_idx] = op((rho*(ut*A1neg - B1)), shared[thread_idx])

    temp1 = pr_by_rho + ut*ut
    temp2 = temp1*A1neg - ut*B1
    shared[thread_idx + block_dim] = op((rho*temp2), shared[thread_idx + block_dim])
    temp1 = ut*un*A1neg - un*B1
    shared[thread_idx + block_dim * 2] = op((rho*temp1), shared[thread_idx + block_dim * 2])
    temp1 = (7*pr_by_rho) + u_sqr
    temp2 = 0.5*ut*temp1*A1neg
    temp1 = (6*pr_by_rho) + u_sqr
    shared[thread_idx + block_dim * 3] = op((rho*(temp2 - 0.5*temp1*B1)), shared[thread_idx + block_dim * 3])
    return nothing
end

function flux_Gyp_kernel(nx, ny, idx, shared, op::Function, thread_idx, block_dim)
    u1 = shared[thread_idx + block_dim * 4]
    u2 = shared[thread_idx + block_dim * 5]
    rho = shared[thread_idx + block_dim * 6]
    pr = shared[thread_idx + block_dim * 7]

    ut = u1*ny - u2*nx
    un = u1*nx + u2*ny

    beta = 0.5*rho/pr
    S2 = un*CUDAnative.sqrt(beta)
    B2 = 0.5*CUDAnative.exp(-S2*S2)/CUDAnative.sqrt(pi*beta)
    A2pos = 0.5*(1 + CUDAnative.erf(S2))

    pr_by_rho = pr/rho
    u_sqr = ut*ut + un*un

    shared[thread_idx] = op((rho*(un*A2pos + B2)), shared[thread_idx])
    temp1 = pr_by_rho + un*un
    temp2 = temp1*A2pos + un*B2
    temp1 = ut*un*A2pos + ut*B2
    #TODO - Verify this is correct
    shared[thread_idx + block_dim] = op((rho*temp1), shared[thread_idx + block_dim])
    shared[thread_idx + block_dim * 2] = op((rho*temp2), shared[thread_idx + block_dim * 2])

    temp1 = (7*pr_by_rho) + u_sqr
    temp2 = 0.5*un*temp1*A2pos
    temp1 = (6*pr_by_rho) + u_sqr
    shared[thread_idx + block_dim * 3] = op((rho*(temp2 + 0.5*temp1*B2)), shared[thread_idx + block_dim * 3])
    return nothing
end

function flux_Gyn_kernel(nx, ny, idx, shared, op::Function, thread_idx, block_dim)
    u1 = shared[thread_idx + block_dim * 4]
    u2 = shared[thread_idx + block_dim * 5]
    rho = shared[thread_idx + block_dim * 6]
    pr = shared[thread_idx + block_dim * 7]

    ut = u1*ny - u2*nx
    un = u1*nx + u2*ny

    beta = 0.5*rho/pr
    S2 = un*CUDAnative.sqrt(beta)
    B2 = 0.5*CUDAnative.exp(-S2*S2)/CUDAnative.sqrt(pi*beta)
    A2neg = 0.5*(1 - CUDAnative.erf(S2))

    pr_by_rho = pr/rho
    u_sqr = ut*ut + un*un

    shared[thread_idx] = op((rho*(un*A2neg - B2)), shared[thread_idx])
    temp1 = pr_by_rho + un*un
    temp2 = temp1*A2neg - un*B2
    temp1 = ut*un*A2neg - ut*B2
    #TODO - Verify this is correct
    shared[thread_idx + block_dim] = op((rho*temp1), shared[thread_idx + block_dim])
    shared[thread_idx + block_dim * 2] = op((rho*temp2), shared[thread_idx + block_dim * 2])

    temp1 = (7*pr_by_rho) + u_sqr
    temp2 = 0.5*un*temp1*A2neg
    temp1 = (6*pr_by_rho) + u_sqr
    shared[thread_idx + block_dim * 3] = op((rho*(temp2 - 0.5*temp1*B2)), shared[thread_idx + block_dim * 3])

    return nothing
end
