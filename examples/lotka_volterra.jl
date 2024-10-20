using DifferentialEquations, DualArrays, Plots

f(u, p, t) = p * u

T = (0.0, 5.0)
u0 = DualArrays.Dual(1., [0.])

d = -1

prob = ODEProblem(f, u0, T, d)
sol = solve(prob)

