# Arithmetic operations for DualArrays.jl

"""
Addition/Subtraction of DualVectors.
"""

for op in (:+, :-)
    @eval Base.$op(x::DualVector, y::DualVector) = DualVector(Base.$op(x.value, y.value), Base.$op(x.jacobian, y.jacobian))
    @eval Base.$op(x::DualVector, y::AbstractVector) = DualVector(Base.$op(x.value, y), x.jacobian)
    @eval Base.$op(x::AbstractVector, y::DualVector) = DualVector(Base.$op(x, y.value), y.jacobian)
end

"""
Matrix multiplication with a DualVector.
"""
Base.:*(x::AbstractMatrix, y::DualVector) = DualVector(x * y.value, x * y.jacobian)

"""
Broadcasted multiplication of two DualVectors using the product rule.
"""

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
            val = Base.$f.(x.value)
            jac = Diagonal(map($p, x.value)) * x.jacobian
            return DualVector(val, jac)
        end
        @eval Base.$f(x::Dual) = Dual(Base.$f(x.value), vec($p(x.value) * (x.partials')))
    elseif n == 2
        p1, p2 = partials
        @eval function Base.broadcasted(::typeof($f), x::DualVector, y::Real)
            val = Base.$f.(x.value, y)
            jac = $p1.(x.value, y) .* x.jacobian
            return DualVector(val, jac)
        end
        @eval function Base.broadcasted(::typeof($f), x::Real, y::DualVector)
            val = Base.$f.(x, y.value)
            jac = $p2.(x, y.value) .* y.jacobian
            return DualVector(val, jac)
        end
        @eval function Base.broadcasted(::typeof($f), x::DualVector, y::Dual)
            val = Base.$f.(x.value, y.value)
            jac = $p1.(x.value, y.value) .* x.jacobian + $p2.(x.value, y.value) * (y.partials')
            return DualVector(val, jac)
        end
        @eval function Base.broadcasted(::typeof($f), x::Dual, y::DualVector)
            val = Base.$f.(x.value, y.value)
            jac = $p1.(x.value, y.value) * (x.partials') + $p2.(x.value, y.value) .* y.jacobian
            return DualVector(val, jac)
        end
        @eval function Base.broadcasted(::typeof($f), x::DualVector, y::DualVector)
            val = Base.$f.(x.value, y.value)
            jac = $p1.(x.value, y.value) .* x.jacobian + $p2.(x.value, y.value) .* y.jacobian
            return DualVector(val, jac)
        end
        @eval Base.$f(x::Dual, y::Dual) = Dual(Base.$f(x.value, y.value), vec($p1(x.value, y.value) * (x.partials') + $p2(x.value, y.value) * (y.partials')))
        @eval Base.$f(x::Dual, y::Real) = Dual(Base.$f(x.value, y), vec($p1(x.value, y) * (x.partials')))
        @eval Base.$f(x::Real, y::Dual) = Dual(Base.$f(x, y.value), vec($p2(x, y.value) * (y.partials')))
    end
end

# Special cases: integer powers
Base.:^(x::Dual, y::Integer) = Dual(x.value ^ y, y * x.value^(y - 1) * x.partials)
Base.broadcasted(::typeof(^), x::DualVector, y::Integer) = DualVector(x.value .^ y, y * x.value .^ (y - 1) .* x.jacobian)
Base.broadcasted(::typeof(Base.literal_pow), ::typeof(^), x::DualVector, ::Val{y}) where y = DualVector(x.value .^ y, y * x.value .^ (y - 1) .* x.jacobian)
Base.literal_pow(::typeof(^), x::Dual, ::Val{y}) where y = Dual(x.value ^ y, y * x.value^(y - 1) * x.partials)

# inner product
LinearAlgebra.dot(x::DualVector, y::DualVector) = Dual(dot(x.value, y.value), vec(y.value' * x.jacobian + x.value' * y.jacobian))
LinearAlgebra.dot(x::DualVector, y::AbstractVector) = Dual(dot(x.value, y), vec(y' * x.jacobian))
LinearAlgebra.dot(x::AbstractVector, y::DualVector) = Dual(dot(x, y.value), y.jacobian' * x)