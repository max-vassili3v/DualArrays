using LinearAlgebra, DualArrays, Plots

function grad_objective(K, f, d, h, kappa, alpha)
    u = K \ f
    p = K \ (u - d)
    return alpha * f + p
end

function solve_1D(x0, d, h, kappa, alpha, iters = 30)
    fvector = copy(x0)
    n = length(x0)
    K = (kappa / h ^ 2) * Tridiagonal(-ones(n - 1), 2 * ones(n), -ones(n - 1))
    for _ = 1:iters
        grads = grad_objective(K, DualVector(fvector, I(length(fvector))), d, h, kappa, alpha)
        step = grads.jacobian \ grads.value
        fvector -= step 
    end

    return fvector
end

function solve_2D(x0, d, h, kappa, alpha, iters = 30)
    fvector = copy(x0)
    n = isqrt(length(x0))
    T = Tridiagonal(-ones(n - 1), 2 * ones(n), -ones(n - 1))
    K = (kappa / h ^ 2) * (kron(I(n), T) + kron(T, I(n)))
    for _ = 1:iters
        grads = grad_objective(K, DualVector(fvector, I(length(fvector))), d, h, kappa, alpha)
        step = grads.jacobian \ grads.value
        fvector -= step 
    end

    return fvector
end

"""
1D Poisson control problem with known solution.
"""
function plot_solution_1D()

    # setup problem
    kappa = 1
    alpha = 1e-6

    T  = (0, 1)
    h = 0.02
    x = collect(T[1]:h:T[2])
    d = sin.(pi * x[2:end-1])

    # we expect this solution of f
    coeff = (kappa * pi^2) / (1 + alpha * kappa^2 * pi^4)
    fexact = coeff .* sin.(pi .* x[2:end-1])

    f = solve_1D(zeros(length(d)), d, h, kappa, alpha)
    plot(x[2:end-1], f, label="Computed Solution")
    plot!(x[2:end-1], fexact, label="Exact Solution")
end
    
"""
2D Poisson control problem on the unit square with known analytical solution
matching the dolfin-adjoint poisson demo.
"""
function plot_solution_2D()

    # setup problem
    kappa = 1
    alpha = 1e-6

    T = (0, 1)
    h = 0.02
    x = collect(T[1]:h:T[2])[2:end-1]
    n = length(x)

    # Create a n x n grid to calculate desired state and exact solution
    # (analogous to numpy meshgrid)
    X = repeat(reshape(x, :, 1), 1, n)
    Y = repeat(reshape(x, 1, :), n, 1)

    d = (1 / (2 * pi^2)) .* sin.(pi .* X) .* sin.(pi .* Y)

    # analytical optimal control as specified in dolfin-adjoint
    fexact = (1 / (1 + 4 * alpha * pi^4)) .* sin.(pi .* X) .* sin.(pi .* Y)

    f = solve_2D(zeros(n * n), vec(d), h, kappa, alpha)
    F = reshape(f, n, n)

    p1 = heatmap(x, x, F, title="Computed Control", aspect_ratio=1)
    p2 = heatmap(x, x, fexact, title="Exact Control", aspect_ratio=1)
    plot(p1, p2, layout=(1, 2), size=(900, 400))
end