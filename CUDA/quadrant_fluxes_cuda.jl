function flux_quad_GxI_kernel(nx, ny, idx, shared, op::Function, thread_idx, block_dim)
    u1 = shared[thread_idx + block_dim * 4]
    u2 = shared[thread_idx + block_dim * 5]
    rho = shared[thread_idx + block_dim * 6]
    pr = shared[thread_idx + block_dim * 7]

    ut = u1*ny - u2*nx
    un = u1*nx + u2*ny

    beta = 0.5*rho/pr
    sqrt_beta = CUDAnative.sqrt(beta)
    S1 = ut*sqrt_beta
    S2 = un*sqrt_beta
    sqrt_pi_beta = CUDAnative.sqrt(pi*beta)
    B1 = 0.5*CUDAnative.exp(-S1*S1)/sqrt_pi_beta
    B2 = 0.5*CUDAnative.exp(-S2*S2)/sqrt_pi_beta
    A1neg = 0.5*(1.0 - CUDAnative.erf(S1))
    A2neg = 0.5*(1.0 - CUDAnative.erf(S2))

    pr_by_rho = pr/rho
    u_sqr = ut*ut + un*un
    shared[thread_idx] = op((rho*A2neg*(ut*A1neg - B1)), shared[thread_idx])

    temp1 = pr_by_rho + ut*ut
    temp2 = temp1*A1neg-ut*B1
    shared[thread_idx + block_dim] = op((rho*A2neg*temp2), shared[thread_idx + block_dim])

    temp1 = ut*A1neg - B1
    temp2 = un*A2neg - B2
    shared[thread_idx + block_dim * 2] = op((rho*temp1*temp2), shared[thread_idx + block_dim * 2])

    temp1 = (7.0 *pr_by_rho) + u_sqr
    temp2 = 0.5*ut*temp1*A1neg

    temp1 = (6.0 *pr_by_rho) + u_sqr
    temp3 = 0.5*B1*temp1
    temp1 = ut*A1neg - B1
    temp4 = 0.5*rho*un*B2*temp1

    shared[thread_idx + block_dim * 3] = op((rho*A2neg*(temp2 - temp3) - temp4), shared[thread_idx + block_dim * 3])
    return nothing
end

function flux_quad_GxII_kernel(nx, ny, idx, shared, op::Function, thread_idx, block_dim)
    u1 = shared[thread_idx + block_dim * 4]
    u2 = shared[thread_idx + block_dim * 5]
    rho = shared[thread_idx + block_dim * 6]
    pr = shared[thread_idx + block_dim * 7]

    ut = u1*ny - u2*nx
    un = u1*nx + u2*ny

    beta = 0.5*rho/pr

    sqrt_beta = CUDAnative.sqrt(beta)
    S1 = ut*sqrt_beta
    S2 = un*sqrt_beta
    sqrt_pi_beta = CUDAnative.sqrt(pi*beta)
    B1 = 0.5*CUDAnative.exp(-S1*S1)/sqrt_pi_beta
    B2 = 0.5*CUDAnative.exp(-S2*S2)/sqrt_pi_beta
    A1pos = 0.5*(1.0 + CUDAnative.erf(S1))
    A2neg = 0.5*(1.0 - CUDAnative.erf(S2))

    pr_by_rho = pr/rho
    u_sqr = ut*ut + un*un
    shared[thread_idx] = op(rho * A2neg* (ut*A1pos + B1), shared[thread_idx])

    temp1 = pr_by_rho + ut*ut
    temp2 = temp1*A1pos + ut*B1
    shared[thread_idx + block_dim] = op((rho*A2neg*temp2), shared[thread_idx + block_dim])

    temp1 = ut*A1pos + B1
    temp2 = un*A2neg - B2
    shared[thread_idx + block_dim * 2] = op((rho*temp1*temp2), shared[thread_idx + block_dim * 2])

    temp1 = (7 *pr_by_rho) + u_sqr
    temp2 = 0.5*ut*temp1*A1pos

    temp1 = (6 *pr_by_rho) + u_sqr
    temp3 = 0.5*B1*temp1

    temp1 = ut*A1pos + B1
    temp4 = 0.5*rho*un*B2*temp1

    shared[thread_idx + block_dim * 3] = op((rho*A2neg*(temp2 + temp3) - temp4), shared[thread_idx + block_dim * 3])
    return nothing
end

function flux_quad_GxIII_kernel(nx, ny, idx, shared, op::Function, thread_idx, block_dim)

    u1 = shared[thread_idx + block_dim * 4]
    u2 = shared[thread_idx + block_dim * 5]
    rho = shared[thread_idx + block_dim * 6]
    pr = shared[thread_idx + block_dim * 7]
    ut = u1*ny - u2*nx
    un = u1*nx + u2*ny

    beta = 0.5*rho/pr
    sqrt_beta = CUDAnative.sqrt(beta)
    S1 = ut*sqrt_beta
    S2 = un*sqrt_beta
    sqrt_pi_beta = CUDAnative.sqrt(pi*beta)
    B1 = 0.5*CUDAnative.exp(-S1*S1)/sqrt_pi_beta
    B2 = 0.5*CUDAnative.exp(-S2*S2)/sqrt_pi_beta
    A1pos = 0.5*(1.0 + CUDAnative.erf(S1))
    A2pos = 0.5*(1.0 + CUDAnative.erf(S2))

    pr_by_rho = pr/rho
    u_sqr = ut*ut + un*un
    shared[thread_idx] = op(rho*A2pos*(ut*A1pos + B1), shared[thread_idx])

    temp1 = pr_by_rho + ut*ut
    temp2 = temp1*A1pos + ut*B1
    shared[thread_idx + block_dim] = op((rho*A2pos*temp2), shared[thread_idx + block_dim])

    temp1 = ut*A1pos + B1
    temp2 = un*A2pos + B2
    shared[thread_idx + block_dim * 2] = op((rho*temp1*temp2), shared[thread_idx + block_dim * 2])

    temp1 = (7*pr_by_rho) + u_sqr
    temp2 = 0.5*ut*temp1*A1pos
    temp1 = (6*pr_by_rho) + u_sqr
    temp3 = 0.5*B1*temp1
    temp1 = ut*A1pos + B1
    temp4 = 0.5*rho*un*B2*temp1

    shared[thread_idx + block_dim * 3] = op((rho*A2pos*(temp2 + temp3) + temp4), shared[thread_idx + block_dim * 3])
    return nothing
end

function flux_quad_GxIV_kernel(nx, ny, idx, shared, op::Function, thread_idx, block_dim)

    u1 = shared[thread_idx + block_dim * 4]
    u2 = shared[thread_idx + block_dim * 5]
    rho = shared[thread_idx + block_dim * 6]
    pr = shared[thread_idx + block_dim * 7]
    ut = u1*ny - u2*nx
    un = u1*nx + u2*ny

    beta = 0.5*rho/pr

    sqrt_beta = CUDAnative.sqrt(beta)
    S1 = ut*sqrt_beta
    S2 = un*sqrt_beta
    sqrt_pi_beta = CUDAnative.sqrt(pi*beta)
    B1 = 0.5*CUDAnative.exp(-S1*S1)/sqrt_pi_beta
    B2 = 0.5*CUDAnative.exp(-S2*S2)/sqrt_pi_beta
    A1neg = 0.5*(1.0 - CUDAnative.erf(S1))
    A2pos = 0.5*(1.0 + CUDAnative.erf(S2))

    pr_by_rho = pr/rho
    u_sqr = ut*ut + un*un

    shared[thread_idx] = op((rho*A2pos*(ut*A1neg - B1)), shared[thread_idx])

    temp1 = pr_by_rho + ut*ut
    temp2 = temp1*A1neg - ut*B1
    shared[thread_idx + block_dim] = op((rho*A2pos*temp2), shared[thread_idx + block_dim])

    temp1 = ut*A1neg - B1
    temp2 = un*A2pos + B2
    shared[thread_idx + block_dim * 2] = op((rho*temp1*temp2), shared[thread_idx + block_dim * 2])

    temp1 = (7.0*pr_by_rho) + u_sqr
    temp2 = 0.5*ut*temp1*A1neg

    temp1 = (6.0 *pr_by_rho) + u_sqr
    temp3 = 0.5*B1*temp1

    temp1 = ut*A1neg - B1
    temp4 = 0.5*rho*un*B2*temp1

    shared[thread_idx + block_dim * 3] = op((rho*A2pos*(temp2 - temp3) + temp4), shared[thread_idx + block_dim * 3])
    return nothing
end

# print(flux_quad_GxI(1,2,3,4,5,5))
