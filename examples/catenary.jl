using DualArrays, LinearAlgebra, Plots, BenchmarkTools

"""
Solving the Catenary Problem using DualArrays.jl
reference: https://www.chebfun.org/examples/opt/Catenary.html

GOAL: Solve the catenary problem from variational calculus using
gradient descent and DualArrays.jl

The computation of finite differences keeps the Jacobian sparse,
so we can use DualArrays.jl
"""

function L(y, h, alpha, beta)
    """
    Evaluate functional with boundary conditions (alpha, beta)
    By approximating y' using finite differences.
    We evaluate L at the midpoints of the intervals
    using centered differences for better stability.
    """
    y_ext = [alpha; y; beta]
    dy = (y_ext[2:end] - y_ext[1:end-1]) / h
    y_mid = (y_ext[1:end-1] + y_ext[2:end]) / 2
    return y_mid .* sqrt.(1 .+ dy.^2)
end

function learn_catenary(h = 0.1, alpha = cosh(-1), beta = cosh(1), epochs = 2000, lr = 0.01)
    n = Int(2 / h) - 1
    y = ones(n) * (alpha + beta) / 2
    for _ = 1:epochs
        jac = jacobian(y -> L(y, h, alpha, beta), y, id="banded")
        grads = h * sum(jac, dims=1)
        y -= lr * vec(grads)
    end
    return y
end

function plot_solution(h)
    x = collect(-1+h:h:1-h)
    y = learn_catenary()
    plot(x, y, label = "Approximation")
    plot!(x, cosh.(x), label = "Exact Solution")
end

function plot_times()
    hs = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
    ns = Int.(2 ./ hs) .- 1
    dualvector_times = Float64[]

    for (h, n) in zip(hs, ns)
        println("Computing solution with h = $h, n = $n")
        push!(dualvector_times, @belapsed learn_catenary($h, cosh(-1), cosh(1), 2000, 0.01))
    end

    plot(ns, dualvector_times, label="DualArrays")
end