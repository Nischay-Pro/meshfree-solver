import SpecialFunctions

function flux_Gxp(nx, ny, u1, u2, rho, pr)

    Gxp = []

    tx = ny
    ty = -nx

    ut = u1*tx + u2*ty
    un = u1*nx + u2*ny

    beta = 0.5*rho/pr
    S1 = ut*sqrt(beta)
    B1 = 0.5*exp(-S1*S1)/sqrt(pi*beta)
    A1pos = 0.5*(1 + SpecialFunctions.erf(S1))

    pr_by_rho = pr/rho
    u_sqr = ut*ut + un*un

    push!(Gxp,(rho*(ut*A1pos + B1)))

    temp1 = pr_by_rho + ut*ut
    temp2 = temp1*A1pos + ut*B1
    push!(Gxp,(rho*temp2))

    temp1 = ut*un*A1pos + un*B1
    push!(Gxp,(rho*temp1))

    temp1 = (7*pr_by_rho) + u_sqr
    temp2 = 0.5*ut*temp1*A1pos
    temp1 = (6*pr_by_rho) + u_sqr
    push!(Gxp,(rho*(temp2 + 0.5*temp1*B1)))
    return Gxp
end

function flux_Gxn(nx, ny, u1, u2, rho, pr)

    Gxn = []

    tx = ny
    ty = -nx

    ut = u1*tx + u2*ty
    un = u1*nx + u2*ny

    beta = 0.5*rho/pr
    S1 = ut*sqrt(beta)
    B1 = 0.5*exp(-S1*S1)/sqrt(pi*beta)
    A1neg = 0.5*(1 - SpecialFunctions.erf(S1))

    pr_by_rho = pr/rho
    u_sqr = ut*ut + un*un

    push!(Gxn,(rho*(ut*A1neg - B1)))

    temp1 = pr_by_rho + ut*ut
    temp2 = temp1*A1neg - ut*B1
    push!(Gxn,(rho*temp2))

    temp1 = ut*un*A1neg - un*B1
    push!(Gxn,(rho*temp1))

    temp1 = (7*pr_by_rho) + u_sqr
    temp2 = 0.5*ut*temp1*A1neg
    temp1 = (6*pr_by_rho) + u_sqr
    push!(Gxn,(rho*(temp2 - 0.5*temp1*B1)))

    return Gxn
end

function flux_Gyp(nx, ny, u1, u2, rho, pr)

    Gyp = []

    tx = ny
    ty = -nx

    ut = u1*tx + u2*ty
    un = u1*nx + u2*ny

    beta = 0.5*rho/pr
    S2 = un*sqrt(beta)
    B2 = 0.5*exp(-S2*S2)/sqrt(pi*beta)
    A2pos = 0.5*(1 + SpecialFunctions.erf(S2))

    pr_by_rho = pr/rho
    u_sqr = ut*ut + un*un

    push!(Gyp,(rho*(un*A2pos + B2)))

    temp1 = pr_by_rho + un*un
    temp2 = temp1*A2pos + un*B2

    temp1 = ut*un*A2pos + ut*B2
    push!(Gyp,(rho*temp1))

    push!(Gyp,(rho*temp2))

    temp1 = (7*pr_by_rho) + u_sqr
    temp2 = 0.5*un*temp1*A2pos
    temp1 = (6*pr_by_rho) + u_sqr
    push!(Gyp,(rho*(temp2 + 0.5*temp1*B2)))

    return Gyp
end

function flux_Gyn(nx, ny, u1, u2, rho, pr)

    Gyn = []

    tx = ny
    ty = -nx

    ut = u1*tx + u2*ty
    un = u1*nx + u2*ny

    beta = 0.5*rho/pr
    S2 = un*sqrt(beta)
    B2 = 0.5*exp(-S2*S2)/sqrt(pi*beta)
    A2neg = 0.5*(1 - SpecialFunctions.erf(S2))

    pr_by_rho = pr/rho
    u_sqr = ut*ut + un*un

    push!(Gyn,(rho*(un*A2neg - B2)))

    temp1 = pr_by_rho + un*un
    temp2 = temp1*A2neg - un*B2

    temp1 = ut*un*A2neg - ut*B2
    push!(Gyn,(rho*temp1))

    push!(Gyn,(rho*temp2))

    temp1 = (7*pr_by_rho) + u_sqr
    temp2 = 0.5*un*temp1*A2neg
    temp1 = (6*pr_by_rho) + u_sqr
    push!(Gyn,(rho*(temp2 - 0.5*temp1*B2)))

    return Gyn
end

function flux_Gx(Gx, nx, ny, u1, u2, rho, pr)
    tx = ny
    ty = -nx

    ut = u1*tx + u2*ty
    un = u1*nx + u2*ny

    Gx[1] = rho*ut

    Gx[2] = pr + rho*ut*ut

    Gx[3] = rho*ut*un

    temp1 = 0.5*(ut*ut + un*un)
    rho_e = 2.5*pr + rho*temp1
    Gx[4] = (pr + rho_e)*ut
end

function flux_Gy(Gy, nx, ny, u1, u2, rho, pr)
    tx = ny
    ty = -nx

    ut = u1*tx + u2*ty
    un = u1*nx + u2*ny

    Gy[1] = rho*un

    Gy[2] = rho*ut*un

    Gy[3] = pr + rho*un*un

    temp1 = 0.5*(ut*ut + un*un)
    rho_e = 2.5*pr + rho*temp1
    Gy[4] = (pr + rho_e)*un
end

# print(flux_Gy(flux_Gxn(1, 3, 2, 2, 2, 1),5,2,3,4,3,2))
