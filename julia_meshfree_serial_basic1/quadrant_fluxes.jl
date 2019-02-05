import SpecialFunctions

function flux_quad_GxI(nx, ny, u1, u2, rho, pr)
    G = []
    tx = ny
    ty = -nx
    ut = u1*tx + u2*ty
    un = u1*nx + u2*ny
    beta = 0.5*rho/pr
    S1 = ut*sqrt(beta)
    S2 = un*sqrt(beta)
    B1 = 0.5*exp(-S1*S1)/sqrt(pi*beta)
    B2 = 0.5*exp(-S2*S2)/sqrt(pi*beta)
    A1neg = 0.5*(1 - SpecialFunctions.erf(S1))
    A2neg = 0.5*(1 - SpecialFunctions.erf(S2))
    pr_by_rho = pr/rho
    u_sqr = ut*ut + un*un
    push!(G, (rho*A2neg*(ut*A1neg - B1)))

    temp1 = pr_by_rho + ut*ut
    temp2 = temp1*A1neg - ut*B1
    push!(G, (rho*A2neg*temp2))

    temp1 = ut*A1neg - B1
    temp2 = un*A2neg - B2
    push!(G, (rho*temp1*temp2))

    temp1 = (7*pr_by_rho) + u_sqr
    temp2 = 0.5*ut*temp1*A1neg

    temp1 = (6*pr_by_rho) + u_sqr
    temp3 = 0.5*B1*temp1

    temp1 = ut*A1neg - B1
    temp4 = 0.5*rho*un*B2*temp1

    push!(G, (rho*A2neg*(temp2 - temp3) - temp4))
    return G
end

function flux_quad_GxII(nx, ny, u1, u2, rho, pr)
    G = []

    tx = ny
    ty = -nx

    ut = u1*tx + u2*ty
    un = u1*nx + u2*ny

    beta = 0.5*rho/pr
    S1 = ut*sqrt(beta)
    S2 = un*sqrt(beta)
    B1 = 0.5*exp(-S1*S1)/sqrt(pi*beta)
    B2 = 0.5*exp(-S2*S2)/sqrt(pi*beta)
    A1pos = 0.5*(1 + SpecialFunctions.erf(S1))
    A2neg = 0.5*(1 - SpecialFunctions.erf(S2))

    pr_by_rho = pr/rho
    u_sqr = ut*ut + un*un

    push!(G, (rho*A2neg*(ut*A1pos + B1)))

    temp1 = pr_by_rho + ut*ut
    temp2 = temp1*A1pos + ut*B1
    push!(G, (rho*A2neg*temp2))

    temp1 = ut*A1pos + B1
    temp2 = un*A2neg - B2
    push!(G, (rho*temp1*temp2))

    temp1 = (7*pr_by_rho) + u_sqr
    temp2 = 0.5*ut*temp1*A1pos

    temp1 = (6*pr_by_rho) + u_sqr
    temp3 = 0.5*B1*temp1

    temp1 = ut*A1pos + B1
    temp4 = 0.5*rho*un*B2*temp1

    push!(G, (rho*A2neg*(temp2 + temp3) - temp4))

    return G
end

function flux_quad_GxIII(nx, ny, u1, u2, rho, pr)

    G = []

    tx = ny
    ty = -nx

    ut = u1*tx + u2*ty
    un = u1*nx + u2*ny

    beta = 0.5*rho/pr
    S1 = ut*sqrt(beta)
    S2 = un*sqrt(beta)
    B1 = 0.5*exp(-S1*S1)/sqrt(pi*beta)
    B2 = 0.5*exp(-S2*S2)/sqrt(pi*beta)
    A1pos = 0.5*(1 + SpecialFunctions.erf(S1))
    A2pos = 0.5*(1 + SpecialFunctions.erf(S2))

    pr_by_rho = pr/rho
    u_sqr = ut*ut + un*un
    push!(G, (rho*A2pos*(ut*A1pos + B1)))

    temp1 = pr_by_rho + ut*ut
    temp2 = temp1*A1pos + ut*B1
    push!(G, (rho*A2pos*temp2))

    temp1 = ut*A1pos + B1
    temp2 = un*A2pos + B2
    push!(G, (rho*temp1*temp2))

    temp1 = (7*pr_by_rho) + u_sqr
    temp2 = 0.5*ut*temp1*A1pos

    temp1 = (6*pr_by_rho) + u_sqr
    temp3 = 0.5*B1*temp1

    temp1 = ut*A1pos + B1
    temp4 = 0.5*rho*un*B2*temp1

    push!(G, (rho*A2pos*(temp2 + temp3) + temp4))

    return G
end

function flux_quad_GxIV(nx, ny, u1, u2, rho, pr)

    G = []

    tx = ny
    ty = -nx

    ut = u1*tx + u2*ty
    un = u1*nx + u2*ny

    beta = 0.5*rho/pr
    S1 = ut*sqrt(beta)
    S2 = un*sqrt(beta)
    B1 = 0.5*exp(-S1*S1)/sqrt(pi*beta)
    B2 = 0.5*exp(-S2*S2)/sqrt(pi*beta)
    A1neg = 0.5*(1 - SpecialFunctions.erf(S1))
    A2pos = 0.5*(1 + SpecialFunctions.erf(S2))

    pr_by_rho = pr/rho
    u_sqr = ut*ut + un*un

    push!(G, (rho*A2pos*(ut*A1neg - B1)))

    temp1 = pr_by_rho + ut*ut
    temp2 = temp1*A1neg - ut*B1
    push!(G, (rho*A2pos*temp2))

    temp1 = ut*A1neg - B1
    temp2 = un*A2pos + B2
    push!(G, (rho*temp1*temp2))

    temp1 = (7*pr_by_rho) + u_sqr
    temp2 = 0.5*ut*temp1*A1neg

    temp1 = (6*pr_by_rho) + u_sqr
    temp3 = 0.5*B1*temp1

    temp1 = ut*A1neg - B1
    temp4 = 0.5*rho*un*B2*temp1
    push!(G, (rho*A2pos*(temp2 - temp3) + temp4))
    return G
end

# print(flux_quad_GxI(1,2,3,4,5,5))
