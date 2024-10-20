using DifferentialEquations, DualArrays, Plots, LinearAlgebra, ForwardDiff

N = 5

function f!(du, u, p, t)
   du[:] = p .* u
end

T = (0.0, 3.0)
u0 = ones(5)

p = -collect(1:N)

prob = ODEProblem(f!, u0, T, p)
sol = solve(prob, saveat = 0.2).u

# k = rand(N)
# lr = 0.01
# for _ = 1:1000
#     probl = ODEProblem(f!, u0, T, k)
#     guess = solve(probl, saveat = 0.2)
#     D = sum([(sol[i] - DualVector(guess.u[i], I(N))) .^ 2 for i = 1:length(guess.t)])
#     grads = sum(D).partials
#     global k -= lr * grads
# end
# k

function solve_eq(k)
    probl = ODEProblem(f!, u0, T, k)
    guess = solve(probl, saveat = 0.2)
    guess
end

