using DualArrays, LinearAlgebra

"""
Solving the Catenary Problem using DualArrays.jl
reference: https://www.chebfun.org/examples/opt/Catenary.html

GOAL: Solve the catenary problem from variational calculus using
gradient descent and DualArrays.jl
"""

function L(y, h, alpha, beta)
    """
    For each element y_i, approximate L(y, y') as
    L(y_i, (y_{i+1} - y_i) / h), returning a vector of L values.
    """
    dy = ([y; beta] .- [alpha; y]) / h
    return y .* sqrt.(1 .+ dy[2:end] .^ 2)
end

function learn_catenary(; h = 0.1, alpha = cosh(-1), beta = cosh(1), epochs = 1000, lr = 0.02)
    y = ones(Int(2/ h) - 1) * (alpha + beta) / 2
    for _ = 1:epochs
        jac = jacobian(y -> L(y, h, alpha, beta), y, id="banded")
        grads = h * sum(jac, dims=1)
        y -= lr * vec(grads)
    end
    return y
end