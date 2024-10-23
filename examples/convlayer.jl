using LinearAlgebra, MLDatasets, Plots, DualArrays, Random, FillArrays, SparseArrays

#GOAL: Implement and differentiate a convolutional neural network layer
function convlayer(img, ker, x, y, xstride = 1, ystride = 1)
    n = size(img, 1) - x + 1
    m = size(img, 2) - y + 1
    flat_img = vcat((sparse_transpose(sparsevec(img[i:i+x-1,j:j+y-1])) for i = 1:xstride:n, j = 1:ystride:m)...)
    flat_img * ker
end

function softmax(x)
    s = sum(exp.(x))
    exp.(x) / s
end

function dense_layer(W, b, x, f::Function = identity)
    ret = W*x
    println("Multiplication complete")
    ret += b
    println("Addition Complete")
    f(ret)
end

function cross_entropy(x, y)
    -sum(y .* log.(x))
end

function model_loss(x, y, w)
    ker = w[1:9]
    weights = reshape(w[10:6769], 10, 676)
    biases = w[6770:6779]
    println("Reshape Complete")
    l1 = convlayer(x, ker, 3, 3)
    println("Conv layer complete")
    l2 = dense_layer(weights, biases, l1, softmax)
    l2
    # println("Dense Layer Complete")
    # target = OneElement(1, y+1, 10)
    # loss = cross_entropy(l2, target)
    # println("Loss complete")
    # loss.value, loss.partials
end

function train_model()
    p = rand(6779)
    epochs = 1
    lr = 0.02
    dataset = MNIST(:train)

    for i = 1:epochs
        train, test = dataset[i]
        d = DualVector(p, I(6779))
        loss = model_loss(sparse(train), test, d)
        return loss
        # println(loss)
        # p = p - lr * grads
    end
end

A = train_model()

