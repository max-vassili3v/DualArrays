using OrdinaryDiffEq, DualArrays, Plots, LinearAlgebra, ForwardDiff

N = 4

function f!(du, u, p, t)
   du[:] = p .* u
end

function lotka_volterra(u, p, t)
    [p[1]*u[1] - p[2]*u[1]*u[2],
    -p[3]*u[2] + p[4]*u[1]*u[2]]
end

T = (0.0, 3.0)
u0 = DualVector(ones(2), zeros(2,2))

p = DualVector(ones(4), I(4))

prob = ODEProblem(lotka_volterra, u0, T, p)
sol = solve(prob, saveat = 0.2).u

# k = rand(N)
# lr = 0.01
# for _ = 1:25
#     probl = ODEProblem(lotka_volterra!, u0, T, k)
#     guess = solve(probl, saveat = 0.2)
#     D = sum([(sol[i] - DualVector(guess.u[i], I(2))) .^ 2 for i = 1:length(guess.t)])
#     grads = sum(D).partials
#     global k -= lr * [grads; grads]
# end
# k

# function solve_eq(k)
#     probl = ODEProblem(f!, u0, T, k)
#     guess = solve(probl, saveat = 0.2)
#     guess
# end

