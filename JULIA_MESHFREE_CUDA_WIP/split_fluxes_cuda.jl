function flux_Gxp_kernel(nx, ny, u1, u2, rho, pr, Gxp1, Gxp2, Gxp3, Gxp4)

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

    Gxp1= (rho*(ut*A1pos + B1))

    temp1 = pr_by_rho + ut*ut
    temp2 = temp1*A1pos + ut*B1
    Gxp2 = (rho*temp2)

    temp1 = ut*un*A1pos + un*B1
    Gxp3 = (rho*temp1)

    temp1 = (7*pr_by_rho) + u_sqr
    temp2 = 0.5*ut*temp1*A1pos
    temp1 = (6*pr_by_rho) + u_sqr
    Gxp4 = (rho*(temp2 + 0.5*temp1*B1))
    return nothing
end

function flux_Gxn_kernel(nx, ny, u1, u2, rho, pr, Gxn1, Gxn2, Gxn3, Gxn4)

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

    Gxn1 = (rho*(ut*A1neg - B1))

    temp1 = pr_by_rho + ut*ut
    temp2 = temp1*A1neg - ut*B1
    Gxn2 = (rho*temp2)

    temp1 = ut*un*A1neg - un*B1
    Gxn3 = (rho*temp1)

    temp1 = (7*pr_by_rho) + u_sqr
    temp2 = 0.5*ut*temp1*A1neg
    temp1 = (6*pr_by_rho) + u_sqr
    Gxn4 = (rho*(temp2 - 0.5*temp1*B1))
    return nothing
end

function flux_Gyp_kernel(nx, ny, u1, u2, rho, pr, Gyp1, Gyp2, Gyp3, Gyp4)
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

    Gyp1 = (rho*(un*A2pos + B2))

    temp1 = pr_by_rho + un*un
    temp2 = temp1*A2pos + un*B2

    temp1 = ut*un*A2pos + ut*B2
    Gyp2 = (rho*temp1)

    Gyp3 = (rho*temp2)

    temp1 = (7*pr_by_rho) + u_sqr
    temp2 = 0.5*un*temp1*A2pos
    temp1 = (6*pr_by_rho) + u_sqr
    Gyp4 = (rho*(temp2 + 0.5*temp1*B2))

    return nothing
end

function flux_Gyn_kernel(nx, ny, u1, u2, rho, pr, Gyn1, Gyn2, Gyn3, Gyn4)
    tx = ny
    ty = -nx

    ut = u1*tx + u2*ty
    un = u1*nx + u2*ny

    beta = 0.5*rho/pr
    S2 = un*CUDAnative.sqrt(beta)
    B2 = 0.5*CUDAnative.exp(-S2*S2)/CUDAnative.sqrt(pi*beta)
    A2neg = 0.5*(1 - CUDAnative.erf(S2))

    pr_by_rho = pr/rho
    u_sqr = ut*ut + un*un

    Gyn1 = (rho*(un*A2neg - B2))

    temp1 = pr_by_rho + un*un
    temp2 = temp1*A2neg - un*B2

    temp1 = ut*un*A2neg - ut*B2
    Gyn2 = (rho*temp1)

    Gyn3 = (rho*temp2)

    temp1 = (7*pr_by_rho) + u_sqr
    temp2 = 0.5*un*temp1*A2neg
    temp1 = (6*pr_by_rho) + u_sqr
    Gyn4 = (rho*(temp2 - 0.5*temp1*B2))

    return nothing
end

function flux_Gx_kernel(Gx1,Gx2,Gx3,Gx4, nx, ny, u1, u2, rho, pr)
    tx = ny
    ty = -nx

    ut = u1*tx + u2*ty
    un = u1*nx + u2*ny

    Gx1 = rho*ut

    Gx2 = pr + rho*ut*ut

    Gx3 = rho*ut*un

    temp1 = 0.5*(ut*ut + un*un)
    rho_e = 2.5*pr + rho*temp1
    Gx4 = (pr + rho_e)*ut
    return nothing
end

function flux_Gy_kernel(Gy1,Gy2,Gy3,Gy4, nx, ny, u1, u2, rho, pr)
    tx = ny
    ty = -nx

    ut = u1*tx + u2*ty
    un = u1*nx + u2*ny

    Gy1 = rho*un

    Gy2 = rho*ut*un

    Gy3 = pr + rho*un*un

    temp1 = 0.5*(ut*ut + un*un)
    rho_e = 2.5*pr + rho*temp1
    Gy4 = (pr + rho_e)*un
    return nothing
end
