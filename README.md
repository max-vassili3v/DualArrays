# DualArrays.jl
DualArrays.jl is a package that provides the `DualVector` structure for use in forward mode automatic differentiation (autodiff). Existing forward mode autodiff implementations such as `ForwardDiff.jl` make use of a `Dual` structure with a vector of dual parts.

There are some limitations of this when differentiating vector valued functions, as the dual components of each element will be each treated as a separate (dense) vector rather than a Jacobian. This misses the opportunity to exploit sparse Jacobian structures such as banded structures that appear when solving systems of ODEs. 

`DualArrays.jl` provides a new structure, `DualVector`, consisting of a vector of real parts and a jacobian that can be any matrix structure in the Julia ecosystem. This carries over many of the optimisations provided by sparse matrix structures to the forward-mode autodiff process. The package also comes with its own `Dual` type for elementwise indexing or vector -> scalar functions. An efficient implementation of differentiating a function might be as follows:

```julia
using FillArrays

function gradient(f::Function, x::Vector)
    dx = DualVector(x, Eye(length(x)))
    return f(dx).jacobian
end
```

See the examples folder for more use cases.

Differentiation rules are mostly provided by the `ChainRules.jl` autodiff backend.
