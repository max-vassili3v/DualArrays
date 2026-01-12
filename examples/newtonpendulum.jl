##
# Solve pendulum ODE:

# x'' + sin(x) = 0

# via discretisation and Newton's method.
##

using LinearAlgebra, ForwardDiff, Plots, DualArrays, FillArrays, BenchmarkTools, BandedMatrices

#Boundary Conditions
a = 0.4
b = 0.0

Tmax = 5.0
ts = 0.01

#LHS of ode
function f(x)
    n = length(x)
    D = Tridiagonal([ones(Float64, n) / ts ; 0.0], [1.0; -2ones(Float64, n) / ts; 1.0], [0.0; ones(Float64, n) / ts])
    (D * [a; x; b])[2:end-1] + sin.(x)
end

#Newton's method using ForwardDiff.jl
function newton_method_forwarddiff(f, x0, n)
    x = x0
    for i = 1:n
        ∇f = ForwardDiff.jacobian(f, x)
        x = x - ∇f \ f(x)
    end
    x
end

#Newton's method using DualArrays.jl
function newton_method_dualvector(f, x0, n)
    x = x0
    l = length(x0)
    for i = 1:n
        ∇f = jacobian(f, x; id=BandedMatrix)
        x = x - ∇f \ f(x)
    end
    x
end

# Plot times for Newton's method using DualArrays and ForwardDiff
function plot_times()
    ns = [10, 50, 100, 200, 500]
    dualvector_times = Float64[]
    forwarddiff_times = Float64[]

    for n in ns
        println("Computing solution with n = $n")
        x0 = zeros(Float64, n)
        push!(dualvector_times, @belapsed newton_method_dualvector(f, $x0, 10))
        push!(forwarddiff_times, @belapsed newton_method_forwarddiff(f, $x0, 10))
    end

    plot(ns, dualvector_times, label="DualArrays")
    plot!(ns, forwarddiff_times, label="ForwardDiff")
end

# Plot solution with obtainded through newtons method with DualArrays.
# Used to verify correctness.
function plot_solution()
    n = Int(Tmax/ts) - 1
    x0 = zeros(Float64, n)
    sol = newton_method_dualvector(f, x0, 10)
    t = ts:ts:(n * ts)
    plot(t, sol, label="Pendulum Solution", xlabel="Time", ylabel="Angle")
end