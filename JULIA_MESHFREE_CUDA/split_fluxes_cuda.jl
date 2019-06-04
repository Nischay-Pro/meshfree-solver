function flux_Gxp_kernel(nx, ny, gpuGlobalDataCommon, gpuGlobalDataRest, idx, flag)

    u1 = gpuGlobalDataCommon[170, idx]
    u2 = gpuGlobalDataCommon[171, idx]
    rho = gpuGlobalDataCommon[172, idx]
    pr = gpuGlobalDataCommon[173, idx]

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
        gpuGlobalDataRest[37, idx]= (rho*(ut*A1pos + B1))

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
        gpuGlobalDataCommon[158, idx] = (rho*(ut*A1pos + B1))

        temp1 = pr_by_rho + ut*ut
        temp2 = temp1*A1pos + ut*B1
        gpuGlobalDataCommon[159, idx] = (rho*temp2)

        temp1 = ut*un*A1pos + un*B1
        gpuGlobalDataCommon[160, idx] = (rho*temp1)

        temp1 = (7*pr_by_rho) + u_sqr
        temp2 = 0.5*ut*temp1*A1pos
        temp1 = (6*pr_by_rho) + u_sqr
        gpuGlobalDataCommon[161, idx] = (rho*(temp2 + 0.5*temp1*B1))
    end
    return nothing
end

function flux_Gxn_kernel(nx, ny, gpuGlobalDataCommon, gpuGlobalDataRest, idx, flag)

    u1 = gpuGlobalDataCommon[170, idx]
    u2 = gpuGlobalDataCommon[171, idx]
    rho = gpuGlobalDataCommon[172, idx]
    pr = gpuGlobalDataCommon[173, idx]

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
        gpuGlobalDataCommon[158, idx] = (rho*(ut*A1neg - B1))

        temp1 = pr_by_rho + ut*ut
        temp2 = temp1*A1neg - ut*B1
        gpuGlobalDataCommon[159, idx] = (rho*temp2)

        temp1 = ut*un*A1neg - un*B1
        gpuGlobalDataCommon[160, idx] = (rho*temp1)

        temp1 = (7*pr_by_rho) + u_sqr
        temp2 = 0.5*ut*temp1*A1neg
        temp1 = (6*pr_by_rho) + u_sqr
        gpuGlobalDataCommon[161, idx] = (rho*(temp2 - 0.5*temp1*B1))
    end
    return nothing
end

function flux_Gyp_kernel(nx, ny, gpuGlobalDataCommon, gpuGlobalDataRest, idx, flag)
    u1 = gpuGlobalDataCommon[170, idx]
    u2 = gpuGlobalDataCommon[171, idx]
    rho = gpuGlobalDataCommon[172, idx]
    pr = gpuGlobalDataCommon[173, idx]

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
        gpuGlobalDataCommon[158, idx] = (rho*(un*A2pos + B2))
        temp1 = pr_by_rho + un*un
        temp2 = temp1*A2pos + un*B2

        temp1 = ut*un*A2pos + ut*B2
        gpuGlobalDataCommon[159, idx] = (rho*temp1)
        gpuGlobalDataCommon[160, idx] = (rho*temp2)
        temp1 = (7*pr_by_rho) + u_sqr
        temp2 = 0.5*un*temp1*A2pos
        temp1 = (6*pr_by_rho) + u_sqr
        gpuGlobalDataCommon[161, idx] = (rho*(temp2 + 0.5*temp1*B2))
    end
    return nothing
end

function flux_Gyn_kernel(nx, ny, gpuGlobalDataCommon, gpuGlobalDataRest, idx, flag)
    u1 = gpuGlobalDataCommon[170, idx]
    u2 = gpuGlobalDataCommon[171, idx]
    rho = gpuGlobalDataCommon[172, idx]
    pr = gpuGlobalDataCommon[173, idx]

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

        gpuGlobalDataCommon[158, idx] = (rho*(un*A2neg - B2))

        temp1 = pr_by_rho + un*un
        temp2 = temp1*A2neg - un*B2

        temp1 = ut*un*A2neg - ut*B2
        gpuGlobalDataCommon[159, idx] = (rho*temp1)

        gpuGlobalDataCommon[160, idx] = (rho*temp2)

        temp1 = (7*pr_by_rho) + u_sqr
        temp2 = 0.5*un*temp1*A2neg
        temp1 = (6*pr_by_rho) + u_sqr
        gpuGlobalDataCommon[161, idx] = (rho*(temp2 - 0.5*temp1*B2))
    end

    return nothing
end

function flux_Gx_kernel(nx, ny, gpuGlobalDataCommon, idx, flag)
    u1 = gpuGlobalDataCommon[170, idx]
    u2 = gpuGlobalDataCommon[171, idx]
    rho = gpuGlobalDataCommon[172, idx]
    pr = gpuGlobalDataCommon[173, idx]

    tx = ny
    ty = -nx

    ut = u1*tx + u2*ty
    un = u1*nx + u2*ny

    if flag == 1
        gpuGlobalDataRest[37, idx] = rho*ut
        gpuGlobalDataRest[38, idx] = pr + rho*ut*ut
        gpuGlobalDataRest[39, idx] = rho*ut*un
        temp1 = 0.5*(ut*ut + un*un)
        rho_e = 2.5*pr + rho*temp1
        gpuGlobalDataRest[40, idx] = (pr + rho_e)*ut
    else
        gpuGlobalDataCommon[158, idx] = rho*ut
        gpuGlobalDataCommon[159, idx] = pr + rho*ut*ut
        gpuGlobalDataCommon[160, idx] = rho*ut*un
        temp1 = 0.5*(ut*ut + un*un)
        rho_e = 2.5*pr + rho*temp1
        gpuGlobalDataCommon[161, idx] = (pr + rho_e)*ut
    end
    return nothing
end

function flux_Gy_kernel(nx, ny,gpuGlobalDataCommon, idx, flag)
    u1 = gpuGlobalDataCommon[170, idx]
    u2 = gpuGlobalDataCommon[171, idx]
    rho = gpuGlobalDataCommon[172, idx]
    pr = gpuGlobalDataCommon[173, idx]

    tx = ny
    ty = -nx

    ut = u1*tx + u2*ty
    un = u1*nx + u2*ny

    if flag == 1
        gpuGlobalDataRest[37, idx] = rho*un
        gpuGlobalDataRest[38, idx] = rho*ut*un
        gpuGlobalDataRest[39, idx] = pr + rho*un*un
        temp1 = 0.5*(ut*ut + un*un)
        rho_e = 2.5*pr + rho*temp1
        gpuGlobalDataRest[40, idx] = (pr + rho_e)*un
    else
        gpuGlobalDataCommon[158, idx] = rho*un
        gpuGlobalDataCommon[159, idx] = rho*ut*un
        gpuGlobalDataCommon[160, idx] = pr + rho*un*un
        temp1 = 0.5*(ut*ut + un*un)
        rho_e = 2.5*pr + rho*temp1
        gpuGlobalDataCommon[161, idx] = (pr + rho_e)*un
    end
    return nothing
end
