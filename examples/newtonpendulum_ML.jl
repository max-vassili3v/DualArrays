##
# Solve pendulum ODE:

# x'' + sin(x) = 0

# via discretisation and Newton's method.
##

using LinearAlgebra, ForwardDiff, Plots, DualArrays, FillArrays, Test, BenchmarkTools

#Boundary Conditions
a = 0.2
b = 0.0

#Time step, Time period and number of x for discretisation.
ts = 0.05

Tmax = 2.0
N = Int(Tmax/ts) - 1

#LHS of ode
function f(x, L)
    n = length(x)
    D = Tridiagonal([ones(n) / ts ; 0.0], [1.0; -2ones(n) / ts; 1.0], [0.0; ones(n) / ts])
    (D * [a; x; b])[2:end-1] + (9.81 / L) * sin.(x)
end


function f(u, a, b, Tmax)
    h = Tmax/(length(u)-1)
    [u[1] - a;
     (u[1:end-2] - 2u[2:end-1] + u[3:end])/h^2 + sin.(u[2:end-1]);
     u[end] - b]
end



#Newtons method using ForwardDiff.jl
function newton_method_forwarddiff(f, x0, n, L)
    x = x0
    for i = 1:n
        ∇f = ForwardDiff.jacobian(y -> f(y, L), x)
        x = x - ∇f \ f(x, L)
    end
    x
end

function newton_method_dualvector(f, x0, n, L)
    x = x0
    l = length(x0)
    for i = 1:n
        ∇f = f(DualVector(x, Eye(l)), L).jacobian
        x = x - ∇f \ f(x, L)
    end
    x
end

function newton_method_dualvector3(f, x0, n, L)
    x = x0
    l = length(x0)
    gr = []
    for i = 1:n
        d = DualVector([x; L], Eye(l + 1))
        jac = f(d[1:l], d[end]).jacobian
        ∇f, gr = jac[:,1:l], jac[:, end]
        x = x - ∇f \ f(x, L)
    end
    x, gr
end

function newton_method_dualvector2(f, x0, n)
    x = x0
    l = length(x0)
    for i = 1:n
        ∇f = f(DualVector(x, Eye(l)), a, b, Tmax).jacobian
        x = x - ∇f \ f(x)
    end
    x
end

#Initial guess
x0 = zeros(N)
L = 10

sol = newton_method_dualvector(f, x0, 100, L)

l = 9.5

lr = 0.1

function learn_length_dualvector(l, n)
    ret = l
    for _ = 1:n
        y, g = newton_method_dualvector3(f, x0, 100, ret)
        d = DualVector(y, reshape(g, :, 1))
        grad = (sum((d - sol).^2) / length(x0)).partials[1]
        ret += lr * grad 
    end
    ret
end

function learn_length_forwarddiff(l, n)
    ret = l
    for _ = 1:n
        d = ForwardDiff.derivative(ret -> (newton_method_forwarddiff(f, x0, 100, ret) - sol).^2 / length(x0), ret)
        ret -= lr * sum(d)
    end
    ret
end

t2 = []
t1 = []

for n = 50:50:500
    println(n)
    push!(t2, @belapsed learn_length_dualvector(9.5, $n))
    push!(t1, @belapsed learn_length_forwarddiff(9.5, $n))
end

