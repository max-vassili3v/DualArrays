# Arithmetic operations for DualArrays.jl

using LinearAlgebra, ChainRules

"""
Addition of DualVectors.
"""
Base.:+(x::DualVector, y::DualVector) = DualVector(x.value + y.value, x.jacobian + y.jacobian)
Base.:+(x::DualVector, y::AbstractVector) = DualVector(x.value + y, x.jacobian)
Base.:+(x::AbstractVector, y::DualVector) = DualVector(x + y.value, y.jacobian)

"""
    *(x::AbstractMatrix, y::DualVector)

Matrix multiplication with a DualVector.
"""
Base.:*(x::AbstractMatrix, y::DualVector) = DualVector(x * y.value, x * y.jacobian)

"""
    broadcasted(::typeof(sin), x::DualVector)

Broadcasted sine function for DualVectors using the chain rule.
"""

Base.broadcasted(::typeof(sin), x::DualVector) = 
    DualVector(sin.(x.value), Diagonal(cos.(x.value)) * x.jacobian)

"""
    broadcasted(::typeof(*), x::DualVector, y::DualVector)

Broadcasted multiplication of two DualVectors using the product rule.
"""
function Base.broadcasted(::typeof(*), x::DualVector, y::DualVector)
    newval = x.value .* y.value
    newjac = Diagonal(x.value) * y.jacobian + Diagonal(y.value) * x.jacobian
    DualVector(newval, newjac)
end

for op in (:-, :/)
    fdef = quote
        function (op)(args::Vararg{Union{DualVector, AbstractVector}, 2})
            _args = (NoTangent(), )
        end
    end
end