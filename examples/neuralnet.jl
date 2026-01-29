##
#
# GOAL: Train a minimal neural network using a
# simple gradient descent algorithm with DualArrays.jl
#
##
using Lux
using ForwardDiff
using Plots
using Random
using DualArrays
using LinearAlgebra

# TODO: Improve to use sparsity

"""
Train a fully connected neural network to learn XOR using DualArrays
"""
function xor_example()
    #Set up training data and neural network in Lux
    data = Dict(
        [0.0, 0.0] => 0.0,
        [0.0, 1.0] => 1.0,
        [1.0, 0.0] => 1.0,
        [1.0, 1.0] => 0.0
    )

    model = Chain(
        Dense(2 => 2, tanh),
        Dense(2 => 1)
    )

    # Initialize parameters and our loss function
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)
    loss = function(y_pred, y_true)
        return (y_pred - y_true)^2
    end

    # Train the model using DualVectors and update with new parameters
    params = train_model(ps, model, loss, data)
    ps = params

    # TESTING: Verify that the model approximates XOR.
    for (feature, label) in data
        y_pred = Lux.apply(model, feature, ps, st)
        println("Input: $feature, Predicted: $y_pred, True: $label")
    end
end

"""
Differentiate the loss of the model with respect to its parameters.
Note: Ideally we would just propagate a NamedTuple of DualVectors
using Lux.apply(), but this does not use sparsity so is inefficient.
See 
"""
function differentiate_model(param, x, lengths, activations, loss)
    # Initialize a dualvector and slice it as follows:
    # subparams[1] = weights of layer 1
    # subparams[2] = biases of layer 1
    # subparams[3] = weights of layer 2
    # etc...
    d = DualVector(param, Matrix(I(length(param))))
    length_ctr = 1
    subparams = []
    for l in lengths
        push!(subparams, d[length_ctr:length_ctr + l - 1])
        length_ctr += l
    end

    # Initialise gradient as input
    grad = x[1]

    for (i, s) in enumerate(subparams)
        # Two iterations correspond to passing through one layer
        # Odd i corresponds to weights
        if i % 2 == 1
            #Get ranges corresponding to each row of the weights matrix
            rows = collect((j:j + length(grad) - 1) for j = 1:length(grad):length(s))
            # Obtain each entry of W * x as a dot product and concatenate them
            grad = vcat([dot(s[r], grad) for r in rows]...)
            # TODO: Do this with a DualMatrix implementation?

        # Even i corresponds to biases. We also apply the activation here.
        else
            # Bias vector
            grad = grad + s
            # Activation function
            grad = activations[Int(i / 2)].(grad)
        end
    end
    grad = loss(sum(grad), x[2])
    return grad
end

"""
Pack the parameter vector into a Lux-style NamedTuple
"""
function pack_params(paramvec, lengths)
    ret = NamedTuple()
    nlayers = Int(length(lengths) / 2)
    keys = Symbol.("layer_" .* string.(1:nlayers))
    values = []
    length_ctr = 1
    for i = 1:nlayers
        weights = paramvec[length_ctr:length_ctr + lengths[2i - 1] - 1]
        length_ctr += lengths[2i - 1]
        bias = paramvec[length_ctr:length_ctr + lengths[2i] - 1]
        length_ctr += lengths[2i]

        weight_matrix = reshape(weights, (:, length(bias)))'
        push!(values, (weight = weight_matrix, bias = bias))
    end
    ret = NamedTuple(k => v for (k, v) in zip(keys, values))
    return ret
end

function train_model(ps, model, loss, x)
    # Flatten parameters into a single vector.
    # We tranpose the weights matrix so that obtaining an entry of W * x is equivalent
    # to a dot product with a slice of the dual vector
    paramvec = vcat([[vec(l.weight'); l.bias] for l in ps]...)
    # We store lengths so we know how to slice the DualVector
    lengths = vcat([[length(vec(l.weight')); length(l.bias)] for l in ps]...)
    # Get activations from model definitions
    activations = [model.layers[i].activation for i=1:length(model.layers)]

    # NOTE: This is a naive gradient descent algorithm. Not guaranteed to converge.
    # TODO: Implement more advanced optimizers in this example.
    curr_loss = 0.0
    for _ = 1:50
        for (feature, label) in x
            grad = differentiate_model(paramvec, (feature, label), lengths, activations, loss)
            curr_loss = grad.value
            paramvec = paramvec - 0.1 * grad.partials
            println("Current Loss: $curr_loss")
        end
        
    end

    # Repack parameters into Lux NamedTuple
    retps = pack_params(paramvec, lengths)
    return retps
end