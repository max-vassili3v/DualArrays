# Indexing operations for DualArrays.jl

using ArrayLayouts, FillArrays, LinearAlgebra, SparseArrays

sparse_getindex(a...) = layout_getindex(a...)
sparse_getindex(D::Diagonal, k::Integer, ::Colon) = OneElement(D.diag[k], k, size(D, 2))
sparse_getindex(D::Diagonal, ::Colon, j::Integer) = OneElement(D.diag[j], j, size(D, 1))

function sparse_getindex(T::Tridiagonal, ::Colon, j::Integer)
    n = size(T, 1)
    if j == 1
        return SparseVector(n, [1,2], [T.d[1], T.dl[1]])
    elseif j == n
        return SparseVector(n, [n-1,n], [T.du[n-1], T.d[n]])
    else
        return SparseVector(n, [j-1,j,j+1], [T.du[j-1], T.d[j], T.dl[j]])
    end
end
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