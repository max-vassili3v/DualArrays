using DifferentialEquations, LinearAlgebra, Plots, BlockArrays, FFMPEG, DualArrays


T = (0.0, 10.0)
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
    sol = solve(prob, abstol = 1e-8, saveat = 0.5)
    sol
end

k = DualVector(rand(N - 1), I(N - 1))

solve_eq(u0, T, k)

