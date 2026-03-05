# Core type definitions for DualArrays.jl

"""
    Dual{T, Partials <: AbstractVector{T}} <: Real

A dual number type that stores a value and its partials (derivatives).

# Fields
- `value::T`: The primal value
- `partials::Partials`: The partial derivatives as a vector
"""
struct Dual{T, Partials <: AbstractVector{T}} <: Real
    value::T
    partials::Partials
end

"""
    DualVector{T, V <: AbstractVector{T}, M <: AbstractMatrix{T}} <: AbstractVector{Dual{T}}

Represents a vector of dual numbers given by:
    
    values + jacobian * [ε₁, …, εₙ]

# Fields
- `value::Vector{T}`: The primal values
- `jacobian::M`: The Jacobian matrix containing partial derivatives

# Constructor
    DualVector(value::Vector{T}, jacobian::M) where {T, M <: AbstractMatrix{T}}

Constructs a DualVector, ensuring that the vector length matches the number of rows in the Jacobian.
"""
struct DualVector{T, V <: AbstractVector{T},M <: AbstractMatrix{T}} <: AbstractVector{Dual{T}}
    value::V
    jacobian::M

    function DualVector(value::V, jacobian::M) where {T, V <: AbstractVector{T}, M <: AbstractMatrix{T}}
        if size(jacobian, 1) != length(value)
            x, y = length(value), size(jacobian, 1)
            throw(ArgumentError("vector length must match number of rows in jacobian.\n" *
                               "vector length: $x\n" *
                               "no. of jacobian rows: $y"))
        end
        new{T,V, M}(value, jacobian)
    end
end

"""
Constructor that forces type compatibility
"""
function DualVector(value::AbstractVector, jacobian::AbstractMatrix)
    T = promote_type(eltype(value), eltype(jacobian))
    DualVector(T.(value), T.(jacobian))
end

"""
DualMatrix{T, M <: AbstractMatrix{T}, A <: AbstractArray{T, 4}} <: AbstractMatrix{Dual{T}}

Represents a matrix of dual numbers given by:

values + jacobian * [ϵ₁₁ ϵ₁₂ … ϵ₁ₘ
                     ϵ₂₁ ϵ₂₂ … ϵ₂ₘ
                     ⋮    ⋮   ⋱   ⋮
                     ϵₙ₁ ϵₙ₂ … ϵₙₘ]

Where the jacobian here is a 4D tensor. This is contracted with with the matrix of ϵs
as follows:

Dᵢⱼ = ∑ₖₗ Jᵢⱼₖₗ ϵₖₗ

Where J is the jacobian tensor, ϵ is the matrix of ϵs, and D is the resulting matrix of
tangents.
"""

struct DualMatrix{T, M <: AbstractMatrix{T}, A <: AbstractArray{T, 4}} <: AbstractMatrix{Dual{T}}
    value::M
    jacobian::A
end

# Basic equality for Dual numbers
==(a::Dual, b::Dual) = a.value == b.value && a.partials == b.partials
isapprox(a::Dual, b::Dual) = isapprox(a.value, b.value) && isapprox(a.partials, b.partials)