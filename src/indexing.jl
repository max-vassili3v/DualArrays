# Indexing operations for DualArrays.jl

using ArrayLayouts, FillArrays, LinearAlgebra, SparseArrays

sparse_getindex(a...) = layout_getindex(a...)
sparse_getindex(d::DualMatrix, a...) = getindex(d, a...)
sparse_getindex(D::Diagonal, k::Integer, ::Colon) = OneElement(D.diag[k], k, size(D, 2))
sparse_getindex(D::Diagonal, ::Colon, j::Integer) = OneElement(D.diag[j], j, size(D, 1))

"""
Extract a single Dual number from a DualVector at position y.
"""
function getindex(x::DualVector, y::Int)
    Dual(x.value[y], sparse_getindex(x.jacobian, y, :))
end

"""
Extract a single Dual number from a DualMatrix at position (y, z).
"""
function getindex(x::DualMatrix, y::Int, z::Int)
    # We collapse the epsilons into a single vector of partials.
    Dual(x.value[y, z], vec(sparse_getindex(x.jacobian, y, z, :, :)))
end

"""
Extract a row slice from a DualMatrix, returning a DualVector.

NOTE: Since a DualVector requires a 2-dimensional Jacobian, and slicing a 4-tensor returns a 3-tensor,
we implicitly 'flatten' the matrix of epsilons. This is done by the reshape.

For example, in a 2x2 case, [ϵ₁₁ ϵ₁₂;ϵ₂₁ ϵ₂₂] becomes [ϵ₁, ϵ₂, ϵ₃, ϵ₄].
"""
function getindex(x::DualMatrix, y::Int, ::Colon)
    n = size(x.value, 2)
    DualVector(x.value[y, :], reshape(sparse_getindex(x.jacobian, y, :, :, :), n, :))
end

"""
Extract a sub-DualVector from a DualVector using a range.
"""
function getindex(x::DualVector, y::UnitRange)
    newval = x.value[y]
    newjac = sparse_getindex(x.jacobian, y, :)
    DualVector(newval, newjac)
end

# Base array interface
for f in (:size, :axes)
    @eval function $f(x::DualArray)
        $f(x.value)
    end
end