"""
    DualArrays

A Julia package for efficient automatic differentiation using dual numbers and dual arrays.

This package provides:
- `Dual`: A dual number type for storing values and their derivatives
- `DualVector`: A vector of dual numbers represented with a Jacobian matrix

Differentiation rules are mostly provided by ChainRules.jl.
"""
module DualArrays

export DualVector, Dual

import Base: +, ==, getindex, size, axes, broadcasted, show, sum, vcat, convert, *

using LinearAlgebra, ArrayLayouts, FillArrays

include("types.jl")
include("indexing.jl")
include("arithmetic.jl")
include("utilities.jl")
end

