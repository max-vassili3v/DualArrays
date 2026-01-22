using DualArrays, StaticArrays, ForwardDiff, BenchmarkTools

n = 200
jacobian = SDiagonal{n, Float64}(ones(n))

dv_static = DualVector(SVector{n, Float64}(rand(n)), jacobian)
v = dv_static.value

dualarraysmethod = function()
    for _ in 1:1000
        tanh.(dv_static)
    end
end

forwarddiffmethod = function()
    for _ in 1:1000
        ForwardDiff.derivative.(tanh, v)
    end
end