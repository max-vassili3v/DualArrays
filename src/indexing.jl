# Indexing operations for DualArrays.jl

using ArrayLayouts, FillArrays, LinearAlgebra, SparseArrays

sparse_getindex(a...) = layout_getindex(a...)
sparse_getindex(D::Diagonal, k::Integer, ::Colon) = OneElement(D.diag[k], k, size(D, 2))
sparse_getindex(D::Diagonal, ::Colon, j::Integer) = OneElement(D.diag[j], j, size(D, 1))

"""
For the purposes of DualArrays.jl, we need to be able to index on the input indices
(corresponding to the the primal values) only. This way, when we index, we still get the
same 'arrangement' of dual parts. Example in the DualVector case (where the Jacobian is
a Tensor{2, Float64, 1, 1} acting on a vector ϵ):

d = a + Jϵ
d[1] = a[1] + J[1,:]ϵ

J[1, :] is a row vector so it can still be applied to our ϵ.
"""
getindex(t::Tensor{L, T, N, M}, i::NTuple{N, Int}, ::Colon) where {L, T, N, M} = sparse_getindex(t.data, i..., ntuple(_ -> Colon(), M)...)
getindex(t::Tensor, i...) = t.data[i...]

"""
Extract a single Dual number from a DualVector at position y.
"""
function Base.getindex(x::DualVector, y::Int)
    Dual(x.value[y], sparse_getindex(x.jacobian, y, :))
end

"""
Extract a sub-DualVector from a DualVector using a range.
"""
function Base.getindex(x::DualVector, y::UnitRange)
    newval = x.value[y]
    newjac = sparse_getindex(x.jacobian, y, :)
    DualVector(newval, newjac)
end

"""
Return the size of the DualVector (length of the value vector).
"""
Base.size(x::DualVector) = (length(x.value),)

"""
Return the axes of the DualVector.
"""
Base.axes(x::DualVector) = axes(x.value)