function flux_Gxp_kernel(nx, ny, gpuGlobalDataRest, idx, shared, flag)

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
    B1 = 0.5*CUDAnative.exp(-S1*S1)/CUDAnative.sqrt(pi*beta)
    A1pos = 0.5*(1 + CUDAnative.erf(S1))

    pr_by_rho = pr/rho
    u_sqr = ut*ut + un*un

    if flag == 1
        gpuGlobalDataRest[37, idx] = (rho*(ut*A1pos + B1))

        temp1 = pr_by_rho + ut*ut
        temp2 = temp1*A1pos + ut*B1
        gpuGlobalDataRest[38, idx] = (rho*temp2)

        temp1 = ut*un*A1pos + un*B1
        gpuGlobalDataRest[39, idx] = (rho*temp1)

        temp1 = (7*pr_by_rho) + u_sqr
        temp2 = 0.5*ut*temp1*A1pos
        temp1 = (6*pr_by_rho) + u_sqr
        gpuGlobalDataRest[40, idx] = (rho*(temp2 + 0.5*temp1*B1))
    else
        gpuGlobalDataRest[41, idx] = (rho*(ut*A1pos + B1))

        temp1 = pr_by_rho + ut*ut
        temp2 = temp1*A1pos + ut*B1
        gpuGlobalDataRest[42, idx] = (rho*temp2)

        temp1 = ut*un*A1pos + un*B1
        gpuGlobalDataRest[43, idx] = (rho*temp1)

        temp1 = (7*pr_by_rho) + u_sqr
        temp2 = 0.5*ut*temp1*A1pos
        temp1 = (6*pr_by_rho) + u_sqr
        gpuGlobalDataRest[44, idx] = (rho*(temp2 + 0.5*temp1*B1))
    end
    return nothing
end

function flux_Gxn_kernel(nx, ny, gpuGlobalDataRest, idx, shared, flag)

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
    B1 = 0.5*CUDAnative.exp(-S1*S1)/CUDAnative.sqrt(pi*beta)
    A1neg = 0.5*(1 - CUDAnative.erf(S1))

    pr_by_rho = pr/rho
    u_sqr = ut*ut + un*un
    if flag == 1
        gpuGlobalDataRest[37, idx] = (rho*(ut*A1neg - B1))

        temp1 = pr_by_rho + ut*ut
        temp2 = temp1*A1neg - ut*B1
        gpuGlobalDataRest[38, idx] = (rho*temp2)

        temp1 = ut*un*A1neg - un*B1
        gpuGlobalDataRest[39, idx] = (rho*temp1)

        temp1 = (7*pr_by_rho) + u_sqr
        temp2 = 0.5*ut*temp1*A1neg
        temp1 = (6*pr_by_rho) + u_sqr
        gpuGlobalDataRest[40, idx] = (rho*(temp2 - 0.5*temp1*B1))
    else
        gpuGlobalDataRest[41, idx] = (rho*(ut*A1neg - B1))

        temp1 = pr_by_rho + ut*ut
        temp2 = temp1*A1neg - ut*B1
        gpuGlobalDataRest[42, idx] = (rho*temp2)

        temp1 = ut*un*A1neg - un*B1
        gpuGlobalDataRest[43, idx] = (rho*temp1)

        temp1 = (7*pr_by_rho) + u_sqr
        temp2 = 0.5*ut*temp1*A1neg
        temp1 = (6*pr_by_rho) + u_sqr
        gpuGlobalDataRest[44, idx] = (rho*(temp2 - 0.5*temp1*B1))
    end
    return nothing
end

function flux_Gyp_kernel(nx, ny, gpuGlobalDataRest, idx, shared, flag)
    u1 = gpuGlobalDataRest[45, idx]
    u2 = gpuGlobalDataRest[46, idx]
    rho = gpuGlobalDataRest[47, idx]
    pr = gpuGlobalDataRest[48, idx]

    tx = ny
    ty = -nx

    ut = u1*tx + u2*ty
    un = u1*nx + u2*ny

    beta = 0.5*rho/pr
    S2 = un*CUDAnative.sqrt(beta)
    B2 = 0.5*CUDAnative.exp(-S2*S2)/CUDAnative.sqrt(pi*beta)
    A2pos = 0.5*(1 + CUDAnative.erf(S2))

    pr_by_rho = pr/rho
    u_sqr = ut*ut + un*un

    if flag == 1
        gpuGlobalDataRest[37, idx] = (rho*(un*A2pos + B2))

        temp1 = pr_by_rho + un*un
        temp2 = temp1*A2pos + un*B2

        temp1 = ut*un*A2pos + ut*B2
        gpuGlobalDataRest[38, idx] = (rho*temp1)
        gpuGlobalDataRest[39, idx] = (rho*temp2)

        temp1 = (7*pr_by_rho) + u_sqr
        temp2 = 0.5*un*temp1*A2pos
        temp1 = (6*pr_by_rho) + u_sqr
        gpuGlobalDataRest[40, idx] = (rho*(temp2 + 0.5*temp1*B2))
    else
        gpuGlobalDataRest[41, idx] = (rho*(un*A2pos + B2))
        temp1 = pr_by_rho + un*un
        temp2 = temp1*A2pos + un*B2

        temp1 = ut*un*A2pos + ut*B2
        gpuGlobalDataRest[42, idx] = (rho*temp1)
        gpuGlobalDataRest[43, idx] = (rho*temp2)
        temp1 = (7*pr_by_rho) + u_sqr
        temp2 = 0.5*un*temp1*A2pos
        temp1 = (6*pr_by_rho) + u_sqr
        gpuGlobalDataRest[44, idx] = (rho*(temp2 + 0.5*temp1*B2))
    end
    return nothing
end

function flux_Gyn_kernel(nx, ny, gpuGlobalDataRest, idx, shared, flag)
    u1 = gpuGlobalDataRest[45, idx]
    u2 = gpuGlobalDataRest[46, idx]
    rho = gpuGlobalDataRest[47, idx]
    pr = gpuGlobalDataRest[48, idx]

    tx = ny
    ty = -nx

    ut = u1*tx + u2*ty
    un = u1*nx + u2*ny

    beta = 0.5*rho/pr
    S2 = un*CUDAnative.sqrt(beta)
    B2 = 0.5*CUDAnative.exp(-S2*S2)/CUDAnative.sqrt(pi*beta)
    A2neg = 0.5*(1 - CUDAnative.erf(S2))


    if flag == 1
        pr_by_rho = pr/rho
        u_sqr = ut*ut + un*un

        gpuGlobalDataRest[37, idx] = (rho*(un*A2neg - B2))

        temp1 = pr_by_rho + un*un
        temp2 = temp1*A2neg - un*B2

        temp1 = ut*un*A2neg - ut*B2
        gpuGlobalDataRest[38, idx] = (rho*temp1)

        gpuGlobalDataRest[39, idx] = (rho*temp2)

        temp1 = (7*pr_by_rho) + u_sqr
        temp2 = 0.5*un*temp1*A2neg
        temp1 = (6*pr_by_rho) + u_sqr
        gpuGlobalDataRest[40, idx] = (rho*(temp2 - 0.5*temp1*B2))
    else
        pr_by_rho = pr/rho
        u_sqr = ut*ut + un*un

        gpuGlobalDataRest[41, idx] = (rho*(un*A2neg - B2))

        temp1 = pr_by_rho + un*un
        temp2 = temp1*A2neg - un*B2

        temp1 = ut*un*A2neg - ut*B2
        gpuGlobalDataRest[42, idx] = (rho*temp1)

        gpuGlobalDataRest[43, idx] = (rho*temp2)

        temp1 = (7*pr_by_rho) + u_sqr
        temp2 = 0.5*un*temp1*A2neg
        temp1 = (6*pr_by_rho) + u_sqr
        gpuGlobalDataRest[44, idx] = (rho*(temp2 - 0.5*temp1*B2))
    end

    return nothing
end
