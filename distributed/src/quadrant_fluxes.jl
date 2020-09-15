import SpecialFunctions

function flux_quad_GxI(G, nx, ny, u1, u2, rho, pr)
    # G = Array{Float64,1}(undef, 0)
    tx = ny
    ty = -nx
    ut = u1*tx + u2*ty
    un = u1*nx + u2*ny
    beta = 0.5*rho/pr
    S1 = ut*sqrt(beta)
    S2 = un*sqrt(beta)
    B1 = 0.5*exp(-S1*S1)/sqrt(Float64(pi)*beta)
    B2 = 0.5*exp(-S2*S2)/sqrt(Float64(pi)*beta)
    A1neg = 0.5*(1.0 - SpecialFunctions.erf(S1))
    A2neg = 0.5*(1.0 - SpecialFunctions.erf(S2))
    pr_by_rho = pr/rho
    u_sqr = ut*ut + un*un
    G[1] = (rho*A2neg*(ut*A1neg - B1))

    temp1 = pr_by_rho + ut*ut
    temp2 = temp1*A1neg-ut*B1
    G[2] = (rho*A2neg*temp2)

    temp1 = ut*A1neg - B1
    temp2 = un*A2neg - B2
    G[3] = (rho*temp1*temp2)

    temp1 = (7.0 *pr_by_rho) + u_sqr
    temp2 = 0.5*ut*temp1*A1neg

    temp1 = (6.0 *pr_by_rho) + u_sqr
    temp3 = 0.5*B1*temp1

    temp1 = ut*A1neg - B1
    temp4 = 0.5*rho*un*B2*temp1

    G[4] = (rho*A2neg*(temp2 - temp3) - temp4)
    return nothing
end

function flux_quad_GxII(G, nx, ny, u1, u2, rho, pr)
    tx = ny
    ty = -nx
    ut = u1*tx + u2*ty
    un = u1*nx + u2*ny

    beta = 0.5*rho/pr
    # if flag == 0
    #     println(IOContext(stdout, :compact => false), rho)
    #     println(IOContext(stdout, :compact => false), pr)
    # end
    S1 = ut*sqrt(beta)
    S2 = un*sqrt(beta)
    B1 = 0.5*exp(-S1*S1)/sqrt(pi*beta)
    B2 = 0.5*exp(-S2*S2)/sqrt(pi*beta)
    A1pos = 0.5*(1.0 + SpecialFunctions.erf(S1))
    A2neg = 0.5*(1.0 - SpecialFunctions.erf(S2))

    pr_by_rho = pr/rho
    u_sqr = ut^2 + un^2
    G[1] = rho * A2neg* (ut*A1pos + B1)

    temp1 = pr_by_rho + ut*ut
    temp2 = temp1*A1pos + ut*B1
    G[2] = rho*A2neg*temp2

    temp1 = ut*A1pos + B1
    temp2 = un*A2neg - B2
    G[3] = rho*temp1*temp2

    # if flag == 0
    #     println("===%%%%===")
    #     println(IOContext(stdout, :compact => false), beta)
    #     println(IOContext(stdout, :compact => false), ut)
    #     println(IOContext(stdout, :compact => false), S1)
    #     println(IOContext(stdout, :compact => false), B1)
    #     println(IOContext(stdout, :compact => false), "####")
    #     println(IOContext(stdout, :compact => false), 0.5* exp(-S1*S1))
    #     println(IOContext(stdout, :compact => false), sqrt(pi*beta))
    #     println(IOContext(stdout, :compact => false), "####")
    #     # println(IOContext(stdout, :compact => false), B2)
    #     println(IOContext(stdout, :compact => false), A1pos)
    #     # println(IOContext(stdout, :compact => false), A2neg)
    #     # println(IOContext(stdout, :compact => false), u_sqr)
    #     println(IOContext(stdout, :compact => false), temp1)
    #     println(IOContext(stdout, :compact => false), temp2)
    #     # println(IOContext(stdout, :compact => false), temp3)
    #     # println(IOContext(stdout, :compact => false), temp4)
    #     println(IOContext(stdout, :compact => false), rho)
    #     println(IOContext(stdout, :compact => false), rho*temp1*temp2)
    #     println()
    # end

    temp1 = (7 *pr_by_rho) + u_sqr
    temp2 = 0.5*ut*temp1*A1pos

    temp1 = (6 *pr_by_rho) + u_sqr
    temp3 = 0.5*B1*temp1

    temp1 = ut*A1pos + B1
    temp4 = 0.5*rho*un*B2*temp1
    G[4] = rho*A2neg*(temp2 + temp3) - temp4

    return nothing
end

function flux_quad_GxIII(G, nx, ny, u1, u2, rho, pr)
    tx = ny
    ty = -nx

    ut = u1*tx + u2*ty
    un = u1*nx + u2*ny

    beta = 0.5*rho/pr
    S1 = ut*sqrt(beta)
    S2 = un*sqrt(beta)
    B1 = 0.5*exp(-S1*S1)/sqrt(pi*beta)
    B2 = 0.5*exp(-S2*S2)/sqrt(pi*beta)
    A1pos = 0.5*(1.0 + SpecialFunctions.erf(S1))
    A2pos = 0.5*(1.0 + SpecialFunctions.erf(S2))

    pr_by_rho = pr/rho
    u_sqr = ut*ut + un*un
    G[1] = (rho*A2pos*(ut*A1pos + B1))

    temp1 = pr_by_rho + ut*ut
    temp2 = temp1*A1pos + ut*B1
    G[2] = (rho*A2pos*temp2)

    temp1 = ut*A1pos + B1
    temp2 = un*A2pos + B2
    G[3] = (rho*temp1*temp2)

    temp1 = (7*pr_by_rho) + u_sqr
    temp2 = 0.5*ut*temp1*A1pos

    temp1 = (6*pr_by_rho) + u_sqr
    temp3 = 0.5*B1*temp1

    temp1 = ut*A1pos + B1
    temp4 = 0.5*rho*un*B2*temp1

    G[4] = (rho*A2pos*(temp2 + temp3) + temp4)

    return nothing
end

function flux_quad_GxIV(G, nx, ny, u1, u2, rho, pr)
    # G = Array{Float64,1}(undef, 0)

    tx = ny
    ty = -nx

    ut = u1*tx + u2*ty
    un = u1*nx + u2*ny

    beta = 0.5*rho/pr
    S1 = ut*sqrt(beta)
    S2 = un*sqrt(beta)
    B1 = 0.5*exp(-S1*S1)/sqrt(pi*beta)
    B2 = 0.5*exp(-S2*S2)/sqrt(pi*beta)
    A1neg = 0.5*(1.0 - SpecialFunctions.erf(S1))
    A2pos = 0.5*(1.0 + SpecialFunctions.erf(S2))

    pr_by_rho = pr/rho
    u_sqr = ut*ut + un*un

    G[1]  = (rho*A2pos*(ut*A1neg - B1))

    temp1 = pr_by_rho + ut*ut
    temp2 = temp1*A1neg - ut*B1
    G[2] =  (rho*A2pos*temp2)

    temp1 = ut*A1neg - B1
    temp2 = un*A2pos + B2
    G[3] =  (rho*temp1*temp2)

    temp1 = (7.0*pr_by_rho) + u_sqr
    temp2 = 0.5*ut*temp1*A1neg

    temp1 = (6.0 *pr_by_rho) + u_sqr
    temp3 = 0.5*B1*temp1

    temp1 = ut*A1neg - B1
    temp4 = 0.5*rho*un*B2*temp1
    G[4] = (rho*A2pos*(temp2 - temp3) + temp4)
    return nothing
end

# print(flux_quad_GxI(1,2,3,4,5,5))
