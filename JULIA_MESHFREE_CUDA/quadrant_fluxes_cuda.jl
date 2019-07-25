function flux_quad_GxI_kernel(nx, ny, gpuGlobalDataRest, idx, shared, op, thread_idx)
    u1 = gpuGlobalDataRest[45, idx]
    u2 = gpuGlobalDataRest[46, idx]
    rho = gpuGlobalDataRest[47, idx]
    pr = gpuGlobalDataRest[48, idx]

    tx = ny
    ty = -nx
    ut = u1*tx + u2*ty
    un = u1*nx + u2*ny
    beta = 0.5*rho/pr
    S1 = ut*CUDAnative.sqrt(beta)
    S2 = un*CUDAnative.sqrt(beta)
    B1 = 0.5*CUDAnative.exp(-S1*S1)/CUDAnative.sqrt(Float64(pi)*beta)
    B2 = 0.5*CUDAnative.exp(-S2*S2)/CUDAnative.sqrt(Float64(pi)*beta)
    A1neg = 0.5*(1.0 - CUDAnative.erf(S1))
    A2neg = 0.5*(1.0 - CUDAnative.erf(S2))

    pr_by_rho = pr/rho
    u_sqr = ut*ut + un*un
    shared[thread_idx + 1] = op((rho*A2neg*(ut*A1neg - B1)), shared[thread_idx + 1])

    temp1 = pr_by_rho + ut*ut
    temp2 = temp1*A1neg-ut*B1
    shared[thread_idx + 2] = op((rho*A2neg*temp2), shared[thread_idx + 2])

    temp1 = ut*A1neg - B1
    temp2 = un*A2neg - B2
    shared[thread_idx + 3] = op((rho*temp1*temp2), shared[thread_idx + 3])

    temp1 = (7.0 *pr_by_rho) + u_sqr
    temp2 = 0.5*ut*temp1*A1neg

    temp1 = (6.0 *pr_by_rho) + u_sqr
    temp3 = 0.5*B1*temp1
    temp1 = ut*A1neg - B1
    temp4 = 0.5*rho*un*B2*temp1

    shared[thread_idx + 4] = op((rho*A2neg*(temp2 - temp3) - temp4), shared[thread_idx + 4])
    return nothing
end

function flux_quad_GxII_kernel(nx, ny, gpuGlobalDataRest, idx, shared, op, thread_idx)
    u1 = gpuGlobalDataRest[45, idx]
    u2 = gpuGlobalDataRest[46, idx]
    rho = gpuGlobalDataRest[47, idx]
    pr = gpuGlobalDataRest[48, idx]

    tx = ny
    ty = -nx
    ut = u1*tx + u2*ty
    un = u1*nx + u2*ny

    beta = 0.5*rho/pr

    S1 = ut*CUDAnative.sqrt(beta)
    S2 = un*CUDAnative.sqrt(beta)
    B1 = 0.5*CUDAnative.exp(-S1*S1)/CUDAnative.sqrt(pi*beta)
    B2 = 0.5*CUDAnative.exp(-S2*S2)/CUDAnative.sqrt(pi*beta)
    A1pos = 0.5*(1.0 + CUDAnative.erf(S1))
    A2neg = 0.5*(1.0 - CUDAnative.erf(S2))

    pr_by_rho = pr/rho
    u_sqr = ut*ut + un*un
    shared[thread_idx + 1] = op(rho * A2neg* (ut*A1pos + B1), shared[thread_idx + 1])

    temp1 = pr_by_rho + ut*ut
    temp2 = temp1*A1pos + ut*B1
    shared[thread_idx + 2] = op((rho*A2neg*temp2), shared[thread_idx + 2])

    temp1 = ut*A1pos + B1
    temp2 = un*A2neg - B2
    shared[thread_idx + 3] = op((rho*temp1*temp2), shared[thread_idx + 3])

    temp1 = (7 *pr_by_rho) + u_sqr
    temp2 = 0.5*ut*temp1*A1pos

    temp1 = (6 *pr_by_rho) + u_sqr
    temp3 = 0.5*B1*temp1

    temp1 = ut*A1pos + B1
    temp4 = 0.5*rho*un*B2*temp1

    shared[thread_idx + 4] = op((rho*A2neg*(temp2 + temp3) - temp4), shared[thread_idx + 4])
    return nothing
end

function flux_quad_GxIII_kernel(nx, ny, gpuGlobalDataRest, idx, shared, op, thread_idx)
    # G = Array{Float64,1}(undef, 0)
    u1 = gpuGlobalDataRest[45, idx]
    u2 = gpuGlobalDataRest[46, idx]
    rho = gpuGlobalDataRest[47, idx]
    pr = gpuGlobalDataRest[48, idx]
    tx = ny
    ty = -nx

    ut = u1*tx + u2*ty
    un = u1*nx + u2*ny

    beta = 0.5*rho/pr
    S1 = ut*CUDAnative.sqrt(beta)
    S2 = un*CUDAnative.sqrt(beta)
    B1 = 0.5*CUDAnative.exp(-S1*S1)/CUDAnative.sqrt(pi*beta)
    B2 = 0.5*CUDAnative.exp(-S2*S2)/CUDAnative.sqrt(pi*beta)
    A1pos = 0.5*(1.0 + CUDAnative.erf(S1))
    A2pos = 0.5*(1.0 + CUDAnative.erf(S2))

    pr_by_rho = pr/rho
    u_sqr = ut*ut + un*un
    shared[thread_idx + 1] = op(rho*A2pos*(ut*A1pos + B1), shared[thread_idx + 1])

    temp1 = pr_by_rho + ut*ut
    temp2 = temp1*A1pos + ut*B1
    shared[thread_idx + 2] = op((rho*A2pos*temp2), shared[thread_idx + 2])

    temp1 = ut*A1pos + B1
    temp2 = un*A2pos + B2
    shared[thread_idx + 3] = op((rho*temp1*temp2), shared[thread_idx + 3])

    temp1 = (7*pr_by_rho) + u_sqr
    temp2 = 0.5*ut*temp1*A1pos
    temp1 = (6*pr_by_rho) + u_sqr
    temp3 = 0.5*B1*temp1
    temp1 = ut*A1pos + B1
    temp4 = 0.5*rho*un*B2*temp1

    shared[thread_idx + 4] = op((rho*A2pos*(temp2 + temp3) + temp4), shared[thread_idx + 4])
    return nothing
end

function flux_quad_GxIV_kernel(nx, ny, gpuGlobalDataRest, idx, shared, op, thread_idx)
    # G = Array{Float64,1}(undef, 0)
    u1 = gpuGlobalDataRest[45, idx]
    u2 = gpuGlobalDataRest[46, idx]
    rho = gpuGlobalDataRest[47, idx]
    pr = gpuGlobalDataRest[48, idx]
    tx = ny
    ty = -nx

    ut = u1*tx + u2*ty
    un = u1*nx + u2*ny

    beta = 0.5*rho/pr
    S1 = ut*CUDAnative.sqrt(beta)
    S2 = un*CUDAnative.sqrt(beta)
    B1 = 0.5*CUDAnative.exp(-S1*S1)/CUDAnative.sqrt(pi*beta)
    B2 = 0.5*CUDAnative.exp(-S2*S2)/CUDAnative.sqrt(pi*beta)
    A1neg = 0.5*(1.0 - CUDAnative.erf(S1))
    A2pos = 0.5*(1.0 + CUDAnative.erf(S2))

    pr_by_rho = pr/rho
    u_sqr = ut*ut + un*un

    shared[thread_idx + 1] = op((rho*A2pos*(ut*A1neg - B1)), shared[thread_idx + 1])

    temp1 = pr_by_rho + ut*ut
    temp2 = temp1*A1neg - ut*B1
    shared[thread_idx + 2] = op((rho*A2pos*temp2), shared[thread_idx + 2])

    temp1 = ut*A1neg - B1
    temp2 = un*A2pos + B2
    shared[thread_idx + 3] = op((rho*temp1*temp2), shared[thread_idx + 3])

    temp1 = (7.0*pr_by_rho) + u_sqr
    temp2 = 0.5*ut*temp1*A1neg

    temp1 = (6.0 *pr_by_rho) + u_sqr
    temp3 = 0.5*B1*temp1

    temp1 = ut*A1neg - B1
    temp4 = 0.5*rho*un*B2*temp1

    shared[thread_idx + 4] = op((rho*A2pos*(temp2 - temp3) + temp4), shared[thread_idx + 4])
    return nothing
end

# print(flux_quad_GxI(1,2,3,4,5,5))
