# Core type definitions for DualArrays.jl

"""
This type represents a linear map from an M-array to an N-array, with a N+M=L-dimensional
array as its underlying data. This can be thought of analogously to an L-tensor equipped
with a contraction pattern characterised by (N, M). Specifically, the tensor has dimensions
a₁ x a₂ x ... x a_N x b₁ x b₂ x ... x b_M and maps an M-array of shape (b₁, b₂, ..., b_M)
to an N-array of shape (a₁, a₂, ..., a_N) by contracting over the last M indices.

We have:
-L is the dimensionality of the tensor
-T is the element type of the tensor
-N is the dimensionality of the input array/number of lower indices
-M is the dimensionality of the output array/number of upper indices

We enforce L = N + M by inferring M in the constructor.

In the context of DualArrays.jl, a DualArray can be thought of as

a + Jϵ 

Where a is an N-array of real numbers, J is an N+M=tensor and ϵ is an M-array of dual parts.
In the simplest case, where N = 0, we have a Dual number with dual parts arranged in an M-array.
"""
struct ArrayOperator{L, T, N, M} <: AbstractArray{T, L}
    data::AbstractArray{T, L}
end

# Constructor to wrap an array with a tensor, given a contraction rule represented by N
function ArrayOperator{N}(data::AbstractArray{T, L}) where {L, T, N}
    ArrayOperator{L, T, N, L - N}(data)
end

# Helper convert function
_convert_array(::Type{T}, a::AbstractArray) where {T} = T.(a)
_convert_array(::Type{T}, t::ArrayOperator{L, S, N, M}) where {T, L, S, N, M} = ArrayOperator{L, T, N, M}(_convert_array(T, t.data))

Base.convert(::Type{ArrayOperator{L, T, N, M}}, tensor::ArrayOperator{L, S, N, M}) where {L, T, N, M, S} =
    ArrayOperator{L, T, N, M}(_convert_array(T, tensor.data))

# Basic array interface
for op in (:size, :axes)
    @eval begin
        ($op)(t::ArrayOperator) = ($op)(t.data)
        ($op)(t::ArrayOperator, i...) = ($op)(t.data, i...)
    end
end

# Below we define a broadcast style for ArrayOperators and override copy and copyto!
# This allows all arithmetic/broadcasting with ArrayOperators to be handled by the
# underlying logic of the array contained in the struct, while ensuring that
# all results stay as a ArrayOperator.

struct ArrayOperatorBroadcastStyle{N} <: Broadcast.AbstractArrayStyle{1} end
ArrayOperatorBroadcastStyle{N}(::Val{M}) where {N, M} = ArrayOperatorBroadcastStyle{N}()

# N is the input dimension of the tensor being broadcasted.
# For he result of the broadcast we will choose to preserve the highest input dimension. 
Base.BroadcastStyle(::Type{<:ArrayOperator{<:Any, <:Any, N, <:Any}}) where {N} = ArrayOperatorBroadcastStyle{N}()
Base.BroadcastStyle(::ArrayOperatorBroadcastStyle{N}, ::Broadcast.DefaultArrayStyle{0}) where {N} = ArrayOperatorBroadcastStyle{N}()
Base.BroadcastStyle(::ArrayOperatorBroadcastStyle{N}, ::Broadcast.DefaultArrayStyle{1}) where {N} = ArrayOperatorBroadcastStyle{N}()
Base.BroadcastStyle(::ArrayOperatorBroadcastStyle{N}, ::ArrayOperatorBroadcastStyle{M}) where {N, M} = ArrayOperatorBroadcastStyle{max(N, M)}()

# Helper functions to help define broadcasting/arithmetic with ArrayOperators.
# By converting a broadcast involving ArrayOperators into a broadcast
# involving the underlying arrays.
_unwrap(t::ArrayOperator) = t.data
_unwrap(x) = x
_unwrap_args(args::Tuple) = map(_unwrap, args)

# copy ensures that arithmetic involving a ArrayOperator returns a ArrayOperator
function Base.copy(bc::Broadcast.Broadcasted{ArrayOperatorBroadcastStyle{N}}) where {N}
    # We create a Broadcasted of the underlying arrays and create a ArrayOperator containing
    # the evaluated broadcast
    databroadcast = Broadcast.Broadcasted(bc.f, _unwrap_args(bc.args), bc.axes)
    ArrayOperator{N}(copy(Broadcast.flatten(databroadcast)))
end

# copyto adds support for .=
function Base.copyto!(dest::ArrayOperator, bc::Broadcast.Broadcasted{ArrayOperatorBroadcastStyle{N}}) where {N}
    # As above
    databroadcast = Broadcast.Broadcasted(bc.f, _unwrap_args(bc.args), bc.axes)
    copyto!(dest.data, Broadcast.flatten(databroadcast))
    dest
end

"""
    Dual{T, Partials <: AbstractVector{T}} <: Real

A dual number type that stores a value and its partials (derivatives).

# Fields
- `value::T`: The primal value
- `partials::Partials`: The partial derivatives as a tensor mapping to a scalar
"""
struct Dual{T, Partials <: (ArrayOperator{L, T, 0, M} where {L, M})} <: Real
    value::T
    partials::Partials
end

function Dual(value::T, partials::ArrayOperator{L, S, 0, M}) where {S, T, L, M}
    T2 = promote_type(T, S)
    Dual{T2, ArrayOperator{L, T2, 0, M}}(convert(T2, value), convert(ArrayOperator{L, T2, 0, M}, partials))
end

# Helper function to define Duals from an AbstractArray
function Dual(value, partials::AbstractArray{T, N}) where {T, N}
    Dual(value, ArrayOperator{0}(partials))
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
struct DualVector{T, V <: AbstractVector{T},J <: (ArrayOperator{L, T, 1, M} where {L, M})} <: AbstractVector{Dual{T}}
    value::V
    jacobian::J

    function DualVector(value::V, jacobian::J) where {T, V <: AbstractVector{T}, J <: (Tensor{L, T, 1, M} where {L, M})}
        if length(value) != size(jacobian, 1)
            throw(ArgumentError("Length of value vector must match number of rows in Jacobian."))
        end
        new{T, V, J}(value, jacobian)
    end
end

"""
Constructor that forces type compatibility
"""
function DualVector(value::AbstractVector, jacobian::ArrayOperator)
    T = promote_type(eltype(value), eltype(jacobian))
    DualVector(_convert_array(T, value), _convert_array(T, jacobian))
end

# Helper function to define DualVectors with AbstractArray jacobians
function DualVector(value::AbstractVector, jacobian::AbstractArray{T, N}) where {T, N}
    DualVector(value, ArrayOperator{1}(jacobian))
end

# Basic equality for Dual numbers
==(a::Dual, b::Dual) = a.value == b.value && a.partials == b.partials
isapprox(a::Dual, b::Dual) = isapprox(a.value, b.value) && isapprox(a.partials, b.partials)