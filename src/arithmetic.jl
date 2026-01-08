# Arithmetic operations for DualArrays.jl

"""
Addition of DualVectors.
"""
Base.:+(x::DualVector, y::DualVector) = DualVector(x.value + y.value, x.jacobian + y.jacobian)
Base.:+(x::DualVector, y::AbstractVector) = DualVector(x.value + y, x.jacobian)
Base.:+(x::AbstractVector, y::DualVector) = DualVector(x + y.value, y.jacobian)

"""
Matrix multiplication with a DualVector.
"""
Base.:*(x::AbstractMatrix, y::DualVector) = DualVector(x * y.value, x * y.jacobian)

"""
Broadcasted multiplication of two DualVectors using the product rule.
"""
function Base.broadcasted(::typeof(*), x::DualVector, y::DualVector)
    newval = x.value .* y.value
    newjac = Diagonal(x.value) * y.jacobian + Diagonal(y.value) * x.jacobian
    DualVector(newval, newjac)
end

"""
this section attempts to define broadcasting rules on DualVectors for functions
that either:

- take a single real argument (function applied to each element)
- are binary operations (binary operation applied to scalar and each element)

We use DiffRules.jl for symbolic derivatives and define our overloads accordingly.
"""

function diff_fn(f, n)
    """
    Helper function: Given an n-ary function f, return its partial derivatives as a tuple of function.
    """
    syms = ntuple(_ -> gensym(), n)
    d = DiffRules.diffrule(:Base, f, syms...)
    d = d isa Tuple ? d : (d,)
    makepartials(dx, syms) = eval(Expr(:->, Expr(:tuple, syms...), dx))
    return map(dx -> makepartials(dx, syms), d)
end

for (_, f, n) in DiffRules.diffrules(filter_modules=(:Base,))
    partials = diff_fn(f, n)
    if n == 1
        p = partials[1]
        @eval function Base.broadcasted(::typeof($f), x::DualVector)
            val = $f.(x.value)
            jac = Diagonal(map($p, x.value)) * x.jacobian
            return DualVector(val, jac)
        end
    elseif n == 2
        p1, p2 = partials
        @eval function Base.broadcasted(::typeof($f), x::DualVector, y::Real)
            val = $f.(x.value, y)
            jac = Diagonal($p1.(x.value, y)) * x.jacobian
            return DualVector(val, jac)
        end
        @eval function Base.broadcasted(::typeof($f), x::Real, y::DualVector)
            val = $f.(x, y.value)
            jac = Diagonal($p2.(x, y.value)) * y.jacobian
            return DualVector(val, jac)
        end
        @eval function Base.broadcasted(::typeof($f), x::DualVector, y::Dual)
            val = $f.(x.value, y.value)
            jac = Diagonal($p1.(x.value, y.value)) * x.jacobian + $p2.(x.value, y.value) * (y.partials')
            return DualVector(val, jac)
        end
        @eval function Base.broadcasted(::typeof($f), x::Dual, y::DualVector)
            val = $f.(x.value, y.value)
            jac = $p1.(x.value, y.value) * (x.partials') + Diagonal($p2.(x.value, y.value)) * y.jacobian
            return DualVector(val, jac)
        end
    end
end