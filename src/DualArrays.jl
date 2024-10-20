module DualArrays

export DualVector, DualArray, DualMatrix, BlockMatrixTensor
export dropzeros

import Base: +, ==, getindex, size, broadcast, axes, broadcasted, show, sum,
             vcat, convert, *, -, ^, /, ndims, hcat, vec, promote_rule, zero,
             reshape, setindex!, one, <, >, <=, >=, log10, decompose
using LinearAlgebra, ArrayLayouts, BandedMatrices, FillArrays, ComponentArrays, SparseArrays
import ChainRules: frule, ZeroTangent

import LinearAlgebra: norm

include("BlockMatrixTensor.jl")
include("DualArray.jl")

end
# module DualArrays