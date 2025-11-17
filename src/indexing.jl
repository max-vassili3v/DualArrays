# Indexing operations for DualArrays.jl

using ArrayLayouts, FillArrays

sparse_getindex(a...) = layout_getindex(a...)
sparse_getindex(D::Diagonal, k::Integer, ::Colon) = OneElement(D.diag[k], k, size(D, 2))
sparse_getindex(D::Diagonal, ::Colon, j::Integer) = OneElement(D.diag[j], j, size(D, 1))

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