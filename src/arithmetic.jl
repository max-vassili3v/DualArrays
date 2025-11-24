# Arithmetic operations for DualArrays.jl

using LinearAlgebra, ChainRules

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

This implementation is loosely based on
https://juliadiff.org/ChainRulesOverloadGeneration.jl/dev/examples/forward_mode.html

"""

"""
Defines how a frule in ChainRules.jl for a scalar function f should be broadcasted over a DualVector.
"""
function broadcast_rule(f, d::DualVector)
    val = similar(d.value)
    jac = similar(d.jacobian)
    jacT = d.jacobian'

    @inbounds for (i, x) in pairs(d.value)
        y, dy = ChainRules.frule((ChainRules.NoTangent(), sparse_getindex(jacT,:, i)), f, x)
        val[i] = y
        fast_setindex!(jac, dy, :, i)
    end

    return DualVector(val, jac')
end

"""
Defines how a frule in ChainRules.jl for a binary
operation a (+) b on reals a, b should be broadcasted over:
- a (+) d (d a DualVector)
- d (+) a (d a DualVector)
"""

function broadcast_rule(f, d::DualVector, x::Real)
    val = similar(d.value)
    jac = similar(d.jacobian)
    z = zero(jac[1, :])

    @inbounds for (i, y) in pairs(d.value)
        yval, dy = ChainRules.frule(
            (ChainRules.NoTangent(), d.jacobian[i, :], z),
             f, y, x)
        val[i] = yval
        jac[i, :] = dy
    end

    return DualVector(val, jac)
end

function broadcast_rule(f, x::Real, d::DualVector)
    val = similar(d.value)
    jac = similar(d.jacobian)
    z = zero(jac[1, :])

    @inbounds for (i, y) in pairs(d.value)
        yval, dy = ChainRules.frule(
            (ChainRules.NoTangent(), z, d.jacobian[i, :]),
             f, x, y)
        val[i] = yval
        jac[i, :] = dy
    end

    return DualVector(val, jac)
end

"""
Extend for broadcasting Dual and DualVector
"""

function broadcast_rule(f, d::DualVector, x::Dual)
    val = similar(d.value)
    jac = similar(d.jacobian)
    z = x.partials

    @inbounds for (i, y) in pairs(d.value)
        yval, dy = ChainRules.frule(
            (ChainRules.NoTangent(), d.jacobian[i, :], z),
             f, y, x.value)
        val[i] = yval
        jac[i, :] = dy
    end

    return DualVector(val, jac)
end

function broadcast_rule(f, x::Dual, d::DualVector)
    val = similar(d.value)
    jac = similar(d.jacobian)
    z = x.partials

    @inbounds for (i, y) in pairs(d.value)
        yval, dy = ChainRules.frule(
            (ChainRules.NoTangent(), z, d.jacobian[i, :]),
             f, x.value, y)
        val[i] = yval
        jac[i, :] = dy
    end

    return DualVector(val, jac)
end

# a set of defined broadcasts to avoid duplicate definitions
defined = Set{DataType}()

# Get all applicable frules defined in ChainRules
# and define broadcasted versions for DualVector using vector_rule

frules = methods(ChainRules.frule)
for f in frules
    # get signatures for each function with a frule
    sig = Base.unwrap_unionall(Base.tuple_type_tail(f.sig))
    
    # split into operation and args, filter for
    # single argument functions and binary operations that can act on real numbers
    op, args = sig.parameters[2], sig.parameters[3:end]

    isconcretetype(op) || continue
    op in defined && continue

    # if it is a single argument function...
    if length(args) == 1
        args[1] isa Type || continue
        Real <: args[1] || continue

        @eval Base.broadcasted(fn::$op, d::DualVector) = broadcast_rule(fn, d)
        push!(defined, op)
    end

    # if it is a binary operation...
    if length(args) == 2
        args[1] isa Type || continue
        args[2] isa Type || continue
        Real <: args[1] && Real <: args[2] || continue

        
        @eval Base.broadcasted(fn::$op, d::DualVector, y::Real) = broadcast_rule(fn, d, y)
        @eval Base.broadcasted(fn::$op, y::Real, d::DualVector) = broadcast_rule(fn, y, d)
        @eval Base.broadcasted(fn::$op, d::DualVector, du::Dual) = broadcast_rule(fn, d, du)
        @eval Base.broadcasted(fn::$op, du::Dual, d::DualVector) = broadcast_rule(fn, du, d)
        push!(defined, op)
    end
end