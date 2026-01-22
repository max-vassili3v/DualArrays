using DualArrays, LinearAlgebra, Plots

"""
Solving the Catenary Problem using DualArrays.jl
reference: https://www.chebfun.org/examples/opt/Catenary.html

GOAL: Solve the catenary problem from variational calculus using
gradient descent and DualArrays.jl
"""

function L(y, h, alpha, beta)
    """
    Evaluate functional with boundary conditions (alpha, beta)
    By approximating y' using finite differences
    """
    y_ext = [alpha; y; beta]
    dy = (y_ext[2:end] - y_ext[1:end-1]) / h
    y_mid = (y_ext[1:end-1] + y_ext[2:end]) / 2
    return y_mid .* sqrt.(1 .+ dy.^2)
end

function learn_catenary(;n = 20, h = 0.1, alpha = cosh(-1), beta = cosh(1), epochs = 10000, lr = 0.01)
    y = ones(n) * (alpha + beta) / 2
    for _ = 1:epochs
        jac = jacobian(y -> L(y, h, alpha, beta), y, id="banded")
        grads = h * sum(jac, dims=1)
        y -= lr * vec(grads)
    end
    return y
end

function plot_solution(n)
    x = LinRange(-1, 1, n)
    y = learn_catenary(n = n)
    plot(x, y, label = "Approximation")
    plot!(x, cosh.(x), label = "Exact Solution")
end