# DualArrays.jl
DualArrays.jl is a package that provides the `DualVector` structure for use in forward mode automatic differentiation (autodiff). Existing forward mode autodiff implementations such as `ForwardDiff.jl` make use of a `Dual` structure representing Dual numbers as follows:

```julia
struct Dual{T,V<:Real,N} <: Real
    value::V
    partials::Partials{N,V}
end
```

where partials is defined as follows:

```julia
struct Partials{N,V} <: AbstractVector{V}
    values::NTuple{N,V}
end
```

There are some limitations of this when differentiating vector valued functions, as having a vector of `Dual`s cannot properly exploit sparse Jacobian structures such as banded structures that appear when solving systems of ODEs. `DualArrays.jl` introduces a new `DualVector` type as follows:

```julia
struct DualVector{T, M <: AbstractMatrix{T}} <: AbstractVector{Dual{T}}
    value::Vector{T}
    jacobian::M
end
```

with a similarly defined `Dual` type. This allows for jacobians to be any sparse matrix structure defined in the Julia linear algebra ecosystem. An efficient implementation of differentiating a function might be as follows:

```julia
using FillArrays

function gradient(f::Function, x::Vector)
    dx = DualVector(x, Eye(length(x)))
    return f(dx).jacobian
end
```

See the examples folder for more use cases
