using OrdinaryDiffEq, LinearAlgebra, Plots, BlockArrays, DualArrays


T = (0.0, 3.0)
N = 8
u0 = [ sin.(collect(0:(N-1)) * pi/(N - 1)); zeros(N)]

function f(u, p, t)
    blocksystem = BlockedArray(zeros(Float64, 2N, 2N), [N, N], [N, N])
    blocksystem[Block(1,2)] = I(N)
    blocksystem[Block(2,1)] = Tridiagonal([p[1:end-1]; 0] , [0; -p[1:end-1]-p[2:end]; 0], [0; p[2:end]])
    parent(blocksystem) * u
end

# For 

# prob = ODEProblem(f, u0, T, k)
# sol = solve(prob, abstol = 1e-8, saveat = 0.5)
# anim = @animate for i in 1:length(sol.t)
#     pos = sol[i][1:N]
#     scatter(1:N, pos, ylim = (-2,2))
# end
# mp4(anim, "demo.mp4", fps = 5)

function solve_eq(u0, T, p)
    prob = ODEProblem(f, u0, T, p)
    sol = solve(prob, abstol = 1e-8, saveat = 0.2)
    sol
end

sol = solve_eq(u0, T, ones(N-1)).u

k = rand(N-1)
lr = 0.0001
for _ = 1:25
    probl = ODEProblem(f, u0, T, k)
    guess = solve(probl, saveat = 0.2)
    D = sum([(sol[i] - DualVector(guess.u[i], I(2N))) .^ 2 for i = 1:length(guess.t)])
    grads = sum(D).partials
    k_grads = [grads[i+1] - grads[i] for i = N+1:2N-1]
    global k -= lr * k_grads
end
k



