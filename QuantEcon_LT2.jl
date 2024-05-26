using LinearAlgebra, Statistics
using LaTeXStrings, QuantEcon, DataFrames, Plots, Random

function ArellanoEconomy(; beta = 0.953,
                         gamma = 2.0,
                         r = 0.017,
                         lambda = 0.9, z = 0.01,
                         rho = 0.945,
                         eta = 0.025,
                         theta = 0.282,
                         ny = 21,
                         nB = 251, nm = 20)

    # create grids
    Bgrid = collect(range(-0.4, 0.4, length = nB))
    mc = tauchen(ny, rho, eta)
    Pi = mc.p
    ygrid = exp.(mc.state_values)
    ydefgrid = min.(0.969 * mean(ygrid), ygrid)

    Pi_m = (1/nm) .* ones(nm, nm)
    m_min = -0.01
    m_max = 0.01
    m_grid = LinRange(m_min, m_max, nm)


    # define value functions
    # notice ordered different than Python to take
    # advantage of column major layout of Julia
    vf = zeros(nB, ny, nm)
    vd = zeros(1, ny, nm)
    vc = zeros(nB, ny, nm)
    policy = zeros(nB, ny, nm)
    q = ones(nB, ny) .* (1 / (1 + r))
    defprob = zeros(nB, ny, nm)

    return (; beta, gamma, r, lambda, z, rho, eta, theta, ny,
            nB, ygrid, ydefgrid, Bgrid, Pi, vf, vd, vc,
            policy, q, defprob, m_grid, nm)
end

u(ae, c) = c^(1 - ae.gamma) / (1 - ae.gamma)

function one_step_update!(ae,
                          EV,
                          EVd,
                          EVc)

    # unpack stuff
    (; beta, gamma, r, lambda, z, rho, eta, theta, ny, nB) = ae
    (; ygrid, ydefgrid, Bgrid, Pi, vf, vd, vc, policy, q, defprob, m_grid, nm) = ae
    zero_ind = searchsortedfirst(Bgrid, 0.0)
    for im in 1:nm
        m = ae.m_grid[im]
        m_min = ae.m_grid[1]
        for iy in 1:ny
            y = ae.ygrid[iy]
            ydef = ae.ydefgrid[iy]
            # value of being in default with income y
            defval = u(ae, ydef + m_min) +
                    beta * (theta * EVc[zero_ind, iy] + (1 - theta) * EVd[1, iy])
            ae.vd[1, iy, im] = defval

            for ib in 1:nB
                B = ae.Bgrid[ib]

                current_max = -1e14
                pol_ind = 0
                for ib_next in 1:nB
                    c = max(y + m - ae.q[ib_next, iy] * (Bgrid[ib_next] - (1-lambda)*B) + (lambda + (1-lambda)*z)*B, 1e-14)
                    mm = u(ae, c) + beta * EV[ib_next, iy]

                    if mm > current_max
                        current_max = mm
                        pol_ind = ib_next
                    end
                end

                # update value and policy functions
                ae.vc[ib, iy, im] = current_max
                ae.policy[ib, iy, im] = Int(pol_ind)
                ae.vf[ib, iy, im] = defval > current_max ? defval : current_max
            end
        end
    end
end


function compute_prices!(ae)
    # unpack parameters
    (; beta, gamma, r, lambda, z, rho, eta, theta, ny, nB) = ae
    (; ygrid, ydefgrid, Bgrid, Pi, vf, vd, vc, policy, q, defprob, m_grid, nm) = ae

    # create default values with a matching size
    vd_compat = repeat(ae.vd, nB, 1, 1)
    default_states = vd_compat .> ae.vc
    
    # update default probabilities and prices
    q_temp = zeros(nB, ny, nm)
    for im in 1:nm
        # ae.defprob[:,:,im] .= default_states[:,:,im] * ae.Pi'
        policies = Int.(ae.policy[:, :, im]) # nB x ny x ny
        # Initialize an array to store the diagonal values of the ny x ny submatrices
        q_new_d = [ae.q[policies[ib, iy], iy] for ib in 1:nB, iy in 1:ny]
        q_new_d = reshape(q_new_d, nB, ny) # Reshape the array to nB x ny
        
        q_temp[:,:,im] .= ( (1 .- default_states[:,:,im]) 
        .* (lambda .+ (1 .- lambda) .* (z .+ q_new_d) ) * ae.Pi' )
    end

    # average over m
    ae.q .= mean(q_temp, dims = 3) / (1 + r)
    
    return
end


function vfi!(ae; tol = 1e-5, maxit = 10000)

    # unpack stuff
    (; beta, gamma, r, lambda, z, rho, eta, theta, ny, nB) = ae
    (; ygrid, ydefgrid, Bgrid, Pi, vf, vd, vc, policy, q, defprob, m_grid, nm) = ae
    Pit = Pi'

    # Iteration stuff
    it = 0
    dist = 10.0

    # allocate memory for update
    V_upd = similar(ae.vf)
    q_upd = similar(ae.q)

    while dist > tol && it < maxit
        it += 1

        # compute expectations for this iterations
        # (we need Pi' because of order value function dimensions)
        copyto!(V_upd, ae.vf)
        copyto!(q_upd, ae.q)
        EV = dropdims(mean(ae.vf, dims = 3), dims = 3) * Pit
        EVd = dropdims(mean(ae.vd, dims = 3), dims = 3) * Pit
        EVc = dropdims(mean(ae.vc, dims = 3), dims = 3) * Pit

        # update value function
        one_step_update!(ae, EV, EVd, EVc)

        # update prices
        compute_prices!(ae)

        dist = maximum(abs(x - y) for (x, y) in zip(V_upd, ae.vf))
        dist2 = maximum(abs(x-y) for (x,y) in zip(q_upd, ae.q))

        if it % 25 == 0
            println("Finished iteration $(it) with dist of $(dist)")
        end
    end
end

ae = ArellanoEconomy(beta = 0.953,      # time discount rate
                     gamma = 2.0,       # risk aversion
                     r = 0.017,         # international interest rate
                     lambda = 0.3, z = 0.01,       
                     rho = 0.945,       # persistence in output
                     eta = 0.025,      # st dev of output shock
                     theta = 0.282,    # prob of regaining access
                     ny = 31,          # number of points in y grid
                     nB = 351,
                     nm = 20)         # number of points in m grid

# now solve the model on the grid.
vfi!(ae)

# create "Y High" and "Y Low" values as 5% devs from mean
high, low = 1.05 * mean(ae.ygrid), 0.95 * mean(ae.ygrid)
iy_high, iy_low = map(x -> searchsortedfirst(ae.ygrid, x), (high, low))

# extract a suitable plot grid
x = zeros(0)
q_low = zeros(0)
q_high = zeros(0)
for i in 1:(ae.nB)
    b = ae.Bgrid[i]
    if -0.35 <= b <= 0  # to match fig 3 of Arellano
        push!(x, b)
        push!(q_low, ae.q[i, iy_low])
        push!(q_high, ae.q[i, iy_high])
    end
end

# generate plot
plot(x, q_low, label = "Low")
plot!(x, q_high, label = "High")
plot!(title = L"Bond price schedule $q(y, B^\prime)$",
      xlabel = L"B^\prime", ylabel = L"q", legend_title = L"y",
      legend = :topleft)

heatmap(ae.Bgrid[1:(end - 1)],
      ae.ygrid[2:end],
      reshape(clamp.(vec( mean(ae.defprob[1:(end - 1), 1:(end - 1),:], dims = 3) ), 0, 1), 350,
              30)')
plot!(xlabel = L"B^\prime", ylabel = L"y", title = "Probability of default",
    legend = :topleft)