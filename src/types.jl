# Core type definitions for DualArrays.jl

"""
This type represents a Tensor along with its contraction rule.
A tensor with its contraction rule represents a linear map from an N-array to an M-array,
with 0-arrays considered scalars, 1-arrays vectors, etc.

We have:
-L is the dimensionality of the tensor
-T is the element type of the tensor
-N is the dimensionality of the input array/number of lower indices
-M is the dimensionality of the output array/number of upper indices

We enforce L = N + M.

In the context of DualArrays.jl, a DualArray (currently only a vector) can be thought of as

a + Jϵ 

Where a is an M-array of real numbers, J is an N+M=tensor and ϵ is an N-array of dual parts.
In the simplest case, where M = 0, we have a Dual number with dual parts arranged in an N-array.
"""
struct Tensor{L, T, N, M} <: AbstractArray{T, L}
    data::AbstractArray{T, L}
end

function Tensor{N,M}(data::AbstractArray{T, L}) where {L, T, N, M}
    if L != N + M
        throw(ArgumentError("Tensor order must be equivalent to N + M."))
    end
    Tensor{L, T, N, M}(data)
end

Base.convert(::Type{Tensor{L, T, N, M}}, tensor::Tensor{L, S, N, M}) where {L, T, N, M, S} =
    Tensor{N, M}(convert.(T, tensor.data))

for op in (:size, :axes)
    @eval begin
        ($op)(t::Tensor) = ($op)(t.data)
    end
end

"""
    Dual{T, Partials <: AbstractVector{T}} <: Real

A dual number type that stores a value and its partials (derivatives).

# Fields
- `value::T`: The primal value
- `partials::Partials`: The partial derivatives as a tensor mapping to a scalar
"""
struct Dual{T, Partials <: (Tensor{L, T, 0, M} where {L, M})} <: Real
    value::T
    partials::Partials
end

function Dual(value::T, partials::Tensor{L, S, 0, M}) where {S, T, L, M}
    T2 = promote_type(T, S)
    Dual{T2, Tensor{L, T2, 0, M}}(convert(T2, value), convert(Tensor{L, T2, 0, M}, partials))
end

# Helper function to define Duals from an AbstractArray
function Dual(value, partials::AbstractArray{T, N}) where {T, N}
    Dual(value, Tensor{0, N}(partials))
end

"""
    DualVector{T, M <: AbstractMatrix{T}} <: AbstractVector{Dual{T}}

Represents a vector of dual numbers given by:
    
    values + jacobian * [ε₁, …, εₙ]

For now the entries just return the values when indexed.

# Fields
- `value::Vector{T}`: The primal values
- `jacobian::M`: The Jacobian matrix containing partial derivatives

# Constructor
    DualVector(value::Vector{T}, jacobian::M) where {T, M <: AbstractMatrix{T}}

Constructs a DualVector, ensuring that the vector length matches the number of rows in the Jacobian.
"""
struct DualVector{T, V <: AbstractVector{T},M <: (Tensor{L, T, 1, M} where {L, M})} <: AbstractVector{Dual{T}}
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
    DualVector(convert(Vector{T}, value), convert(AbstractMatrix{T}, jacobian))
end

# Basic equality for Dual numbers
==(a::Dual, b::Dual) = a.value == b.value && a.partials == b.partials
isapprox(a::Dual, b::Dual) = isapprox(a.value, b.value) && isapprox(a.partials, b.partials)