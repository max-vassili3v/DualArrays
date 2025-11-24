###
# Illustration of computing times for jacobians
# using DualArrays and ForwardDiff
###

using DualArrays, ForwardDiff, Plots, BenchmarkTools

f(x) = sin.(x)

function plot_times()
    dualarray_times = Float64[]
    forwarddiff_times = Float64[]
    ns = [10, 20, 50, 100, 200, 500, 1000]

    for n in ns
        println("Computing jacobian for n = $n")
        push!(dualarray_times, @belapsed DualArrays.jacobian(f, rand($n)))
        push!(forwarddiff_times, @belapsed ForwardDiff.jacobian(f, rand($n)))
    end

    plot(ns, dualarray_times, label="DualArrays")
    plot!(ns, forwarddiff_times, label="ForwardDiff")
end