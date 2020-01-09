using LinearAlgebra, Statistics, Random, Plots
plotly()

# We define some useful activation functions
sigmoid(x) = exp(x)/(1 + exp(x))
relu(x) = max(0,x)
leakyrelu(x) = max(0.2*x,x)

# And methods to calculate their derivatives
derivative(f::typeof(sigmoid), x::Float64) = sigmoid(x)*(1-sigmoid(x))
derivative(f::typeof(identity), x::Float64) = one(x)
derivative(f::typeof(relu), x::Float64) = derivative_relu(x)
derivative(f::typeof(leakyrelu), x::Float64) = derivative_leakyrelu(x)
function derivative_relu(x)
    if x > 0
        tmp = 1
    else
        tmp = 0
    end
end
function derivative_leakyrelu(x)
    if x > 0
        tmp = 1
    else
        tmp = 0.2
    end
end

# Astract type, all layers will be a subtype of `Layer`
abstract type Layer{T} end

""" Dense layer for `σ(W*z+b)`,
    stores the intermediary value z as well as the output, gradients and δ"""
struct Dense{T, F<:Function} <: Layer{T}
    W::Matrix{T}
    b::Vector{T}
    σ::F
    x::Vector{T}    # W*z+b
    out::Vector{T}  # σ(W*z+b)
    ∂W::Matrix{T}   # ∂J/dW
    ∇b::Vector{T}   # (∂J/db)ᵀ
    δ::Vector{T}    # dJ/dz
end

""" layer = Dense(nout, nin, σ::F=sigmoid, W0 = 1.0, Wstd = 0.1, b0=0.0, bstd = 0.1)
    Dense layer for `σ(W*x+b)` with nout outputs and nin inputs, with activation function σ.
    `W0, Wstd, b0, bstd` adjusts the mean and standard deviation of the initial weights. """
function Dense(nout, nin, σ::F=sigmoid, W0 = 1.0, Wstd = 0.1, b0=0.0, bstd = 0.1) where F
    W = W0/nin/nout .+ Wstd/nin/nout .* randn(nout, nin)
    b = b0 .+ bstd.*randn(nout)
    x = similar(b)
    out = similar(x)
    ∂W = similar(W)
    ∇b = similar(x)
    δ = similar(x, nin)
    Dense{Float64, F}(W, b, σ, x, out, ∂W, ∇b, δ)
end

""" out = l(z)
    Compute the output `out` from the layer.
    Store the input to the activation function in l.x and the output in l.out. """
function (l::Dense)(z)
    a = l.W*z + l.b
    for i = 1:length(a)
        l.x[i] = a[i]
        l.out[i] = l.σ(l.x[i])
    end
    return l.out
end

# A network is just a sequence of layers
struct Network{T,N<:Layer{T}}
    layers::Vector{N}
end

""" out = n(z)
    Comute the result of applying each layer in a network to the previous output. """
function (n::Network)(z)
    temp = copy(z)
    for i = 1:length(n.layers)
        temp = n.layers[i](temp)
    end
    return temp
end

""" δ = backprop!(l::Dense, δnext, zin)
    Assuming that layer `l` has been called with `zin`,
    calculate the l.δ = ∂L/∂zᵢ given δᵢ₊₁ and zᵢ,
    and save l.∂W = ∂L/∂Wᵢ and l.∇b = (∂L/∂bᵢ)ᵀ """
function backprop!(l::Dense, δnext, zin)
    sz = size(l.W)
    for i = 1:sz[1]
        l.∇b[i] = δnext[i]*derivative(l.σ,l.x[i])
        for j = 1:sz[2]
            l.∂W[i,j] = l.∇b[i]*zin[j]
        end
    end
    for i = 1:sz[2]
        l.δ[i] = sum(l.W[:,i].*l.∇b)
    end
    #l.∂W = l.∇b*copy(zin')
    #l.δ = copy(l.W')*l.∇b
    return l.δ
end


""" backprop!(n::Network, input, ∂J∂y)
    Assuming that network `n` has been called with `input`, i.e `y=n(input)`
    backpropagate and save all gradients in the network,
    where ∂J∂y is the gradient (∂J/∂y)ᵀ. """
function backprop!(n::Network, input, ∂J∂y)
    layers = n.layers
    # To the last layer, δᵢ₊₁ is ∂J∂y
    δ = ∂J∂y
    # Iterate through layers, starting at the end
    for i in length(layers):-1:2
        for j = 1:length(δ)
            layers[i].δ[j] = δ[j]
        end
        δ = backprop!(layers[i], δ, layers[i-1].out)
        #+++ Fill in the missing code here
        #+++
    end
    # To first layer, the input was `input`
    zin = input
    δ = backprop!(layers[1], δ, zin)
    return
end

function gradientstep!(n, lossfunc, x, y)
    out = n(x)
    # Calculate (∂L/∂out)T
    ∇L = derivative(lossfunc, out, y)
    # Backward pass over network
    backprop!(n, x, ∇L)
    # Get list of all parameters and gradients
    parameters, gradients = getparams(n)
    # For each parameter, take gradient step
    for i = 1:length(parameters)
        p = parameters[i]
        g = gradients[i]
        # Update this parameter with a small step in negative gradient 􏰀→ direction
        p .= p .- 0.001.*g
        # The parameter p is either a W, or b so we broadcast to update all the 􏰀→ elements
    end
end

# This can be used to get a list of all parameters and  from a Dense layer
getparams(l::Dense) = ([l.W, l.b], [l.∂W, l.∇b])

""" `params,  = getparams(n::Network)`
    Return a list of references to all paramaters and corresponding . """
function getparams(n::Network{T}) where T
    params = Array{T}[]         # List of references to vectors and matrices (arrays) of parameters
    gradients = Array{T}[]      # List of references to vectors and matrices (arrays) of
    for layer in n.layers
        p, g = getparams(layer)
        append!(params, p)      # push the parameter references to params list
        append!(gradients, g)   # push the gradient references to  list
    end
    return params, gradients
end

### Define loss function L(y,yhat)
sumsquares(yhat,y) =  norm(yhat-y)^2
# And its gradient with respect to yhat: L_{yhat}(yhat,y)
derivative(::typeof(sumsquares), yhat, y) =  yhat - y

""" Structure for saving all the parameters and states needed for ADAM,
    as well as references to the parameters and  """
struct ADAMTrainer{T,GT}
    n::Network{T}
    β1::T
    β2::T
    ϵ::T
    γ::T
    params::GT              # List of paramaters in the network (all Wᵢ and bᵢ)
    gradients::GT           # List of  (all ∂Wᵢ and ∇bᵢ)
    ms::GT                  # List of mₜ for each parameter
    mhs::GT                 # List of \hat{m}ₜ for each parameter
    vs::GT                  # List of vₜ for each parameter
    vhs::GT                 # List of \hat{v}ₜ for each parameter
    t::Base.RefValue{Int}   # Reference to iteration counter
end

function ADAMTrainer(n::Network{T}, β1 = 0.9, β2 = 0.999, ϵ=1e-8, γ=0.1) where T
    params, gradients = getparams(n)
    ms = [zero(gi) for gi in gradients]
    mhs = [zero(gi) for gi in gradients]
    vs = [ones(size(gi)...) for gi in gradients]
    vhs = [zero(gi) for gi in gradients]
    ADAMTrainer{T, typeof(params)}(n, β1, β2, ϵ, γ, params, gradients, ms, mhs, vs, vhs, Ref(1))
end

""" `update!(At::ADAMTrainer)`
    Assuming that all gradients are already computed using backpropagation,
    take a step with the ADAM algorithm """
function update!(At::ADAMTrainer)
    # Get some of the variables that we need from the ADAMTrainer
    β1, β2, ϵ, γ = At.β1, At.β2, At.ϵ, At.γ
    # At.t is a reference, we get the value t like this
    t = At.t[]
    # For each of the W and b in the network
    for i in eachindex(At.params)
        p = At.params[i]        # This will reference either a W or b
        ∇p = At.gradients[i]    # This will reference either a ∂W or ∇b
        # Get each of the stored values m, mhat, v, vhat for this parameter
        m, mh, v, vh = At.ms[i], At.mhs[i], At.vs[i], At.vhs[i]

        # Update ADAM parameters
        At.ms[i] = β1*m+(1-β1)*∇p
        At.mhs[i] = m/(1-β1^t)
        At.vs[i] = β2*v+(1-β2)*∇p.^2
        At.vhs[i] = v/(1-β2^t)
        At.params[i]  = p - γ*((sqrt.(vh).+ϵ).\mh)   #new sqrt OLD: p - γ*(sqrt.(vh+.ϵ).\mh)
        #+++
        #+++
        #+++
        #+++

        # Take the ADAM step

        #+++
    end
    At.t[] = t+1     # At.t is a reference, we update the value t like this
    return
end


""" `loss = train!(n, alg, xs, ys, lossfunc)`

    Train a network `n` with algorithm `alg` on inputs `xs`, expected outputs `ys`
    for loss-function `lossfunc` """
function train!(n, alg, xs, ys, lossfunc)
    lossall = 0.0           # This will keep track of the sum of the losses

    parameters,_ = getparams(n)
    for i in eachindex(xs)  # For each data point
        xi = xs[i]          # Get data
        yi = ys[i]          # And expected output

        #+++
        #+++
        out = n(xi)
        ∇L = derivative(lossfunc, out, yi)
        backprop!(n, xi, ∇L)
        #adam = ADAMTrainer(n)
        update!(alg)

        for j = 1:length(parameters)
            p = alg.params[j]
            g = alg.gradients[j]
            # Update this parameter with a small step in negative gradient 􏰀→ direction
            parameters[j] .= p .- 0.001.*g
            # The parameter p is either a W, or b so we broadcast to update all the 􏰀→ elements
        end

        # Backward pass over network
        #+++ Do a forward and backwards pass
        #+++ with `xi`, `yi, and
        #+++ update parameters using `alg`
        #+++
        #+++
        #+++

        loss = lossfunc(out, yi)
        lossall += loss
    end
    # Calculate and print avergae loss
    avgloss = lossall/length(xs)
    println("Avg loss: $avgloss")
    return avgloss
end

""" `testloss(n, xs, ys, lossfunc)`
    Evaluate mean loss of network `n`, over data `xs`, `ys`,
    using lossfunction `lossfunc` """
getloss(n, xs, ys, lossfunc) = mean(xy -> lossfunc(xy[2], n(xy[1])), zip(xs,ys))


#########################################################
#########################################################
#########################################################
### Task 3:

### Define network
# We use some reasonable value on initial weights
l1 = Dense(30, 1, leakyrelu, 0.0, 3.0, 0.0, 0.1)
lis = [Dense(30, 30, leakyrelu, 0.0, 3.0, 0.0, 0.1) for i = 1:4]
# Last layer has no activation function (identity)
ln = Dense(1, 30, identity, 0.0, 1.0, 0.0, 0.1)
n = Network([l1, lis..., ln])

### This is the function we want to approximate
fsol(x) = [min(3,norm(x)^2)]

### Define data, in range [-4,4]
xs = [rand(1).*8 .- 4 for i = 1:2000] #30 for task 5 and 2000 for task 4 and 6
ys = [fsol(xi) for xi in xs]

# Test data
testxs = [rand(1).*8 .- 4 for i = 1:1000]
testys = [fsol(xi) for xi in testxs]

### Define algorithm
adam = ADAMTrainer(n, 0.95, 0.999, 1e-8, 0.0001)

### Train and plot

# Train once over the data set
@time train!(n, adam, xs, ys, sumsquares)
scatter(xs, [copy(n(xi)) for xi in xs], legend =:false)

# Train 100 times over the data set
for i = 1:100
    # Random ordering of all the data
    Iperm = randperm(length(xs))
    @time train!(n, adam, xs[Iperm], ys[Iperm], sumsquares)
end

# Plot real line and prediction
plot(-4:0.01:4, [fsol.(xi)[1] for xi in -4:0.01:4], c=:blue, legend=:false)
scatter!(xs, ys, lab="", m=(:cross,0.2,:blue), legend=:false)
scatter!(xs, [copy(n(xi)) for xi in xs], m=(:circle,0.2,:red), legend=:false)

# We can calculate the mean error over the training data like this also
getloss(n, xs, ys, sumsquares)
# Loss over test data like this
getloss(n, testxs, testys, sumsquares)
print("Mean error over training; ",string(getloss(n, xs, ys, sumsquares)),"\n","Mean error over test; ",string(getloss(n, testxs, testys, sumsquares)))

# Plot expected line
plot(-8:0.01:8, [fsol.(xi)[1] for xi in -8:0.01:8], c=:blue, label="Function",legend=:bottomright);
# Plot full network result
plot!(-8:0.01:8, [copy(n([xi]))[1] for xi in -8:0.01:8], c=:red, label="Network values", legend=:bottomright)

#########################################################
#########################################################
#########################################################
### Task 4:
l1 = Dense(30, 1, leakyrelu, 0.0, 3.0, 0.0, 0.1)
lis = [Dense(30, 30, leakyrelu, 0.0, 3.0, 0.0, 0.1) for i = 1:4]
# Last layer has no activation function (identity)
ln = Dense(1, 30, identity, 0.0, 1.0, 0.0, 0.1)
n = Network([l1, lis..., ln])

fsol(x) = [min(3,norm(x)^2)]

### Define data, in range [-4,4]
xs = [rand(1).*8 .- 4 for i = 1:2000] #30 for task 5 and 2000 for task 4 and 6
ys = [fsol(xi).+ 0.1.*randn(1) for xi in xs]

# Test data
testxs = [rand(1).*8 .- 4 for i = 1:1000]
testys = [fsol(xi) for xi in testxs]

### Define algorithm
adam = ADAMTrainer(n, 0.95, 0.999, 1e-8, 0.0001)

@time train!(n, adam, xs, ys, sumsquares)
scatter(xs, [copy(n(xi)) for xi in xs])

# Train 100 times over the data set
for i = 1:100
    # Random ordering of all the data
    Iperm = randperm(length(xs))
    @time train!(n, adam, xs[Iperm], ys[Iperm], sumsquares)
end

# Plot real line and prediction
plot(-4:0.01:4, [fsol.(xi)[1] for xi in -4:0.01:4], c=:red, lab="",legend =:bottomright)
scatter!(xs, ys, lab="", m=(:cross,0.2,:blue))
scatter!(xs, [copy(n(xi)) for xi in xs], m=(:circle,0.2,:red),lab="")

# We can calculate the mean error over the training data like this also
getloss(n, xs, ys, sumsquares)
# Loss over test data like this
getloss(n, testxs, testys, sumsquares)
print("Mean error over training; ",string(getloss(n, xs, ys, sumsquares)),"\n","Mean error over test; ",string(getloss(n, testxs, testys, sumsquares)))

plot(-8:0.01:8, [fsol.(xi)[1] for xi in -8:0.01:8], c=:blue, label="Function",legend=:bottomright);
# Plot full network result
plot!(-8:0.01:8, [copy(n([xi]))[1] for xi in -8:0.01:8], c=:red, label="Network values", legend=:bottomright)

#########################################################
#########################################################
#########################################################
### Task 5:
l1 = Dense(30, 1, leakyrelu, 0.0, 3.0, 0.0, 0.1)
lis = [Dense(30, 30, leakyrelu, 0.0, 3.0, 0.0, 0.1) for i = 1:4]
# Last layer has no activation function (identity)
ln = Dense(1, 30, identity, 0.0, 1.0, 0.0, 0.1)
n = Network([l1, lis..., ln])

fsol(x) = [min(3,norm(x)^2)]

### Define data, in range [-4,4]
xs = [rand(1).*8 .- 4 for i = 1:30] #30 for task 5 and 2000 for task 4 and 6
ys = [fsol(xi).+ 0.1.*randn(1) for xi in xs]

# Test data
testxs = [rand(1).*8 .- 4 for i = 1:1000]
testys = [fsol(xi) for xi in testxs]

### Define algorithm
adam = ADAMTrainer(n, 0.95, 0.999, 1e-8, 0.0001)

@time train!(n, adam, xs, ys, sumsquares)
scatter(xs, [copy(n(xi)) for xi in xs])

# Train 10000 times over the data set
for i = 1:5000
    # Random ordering of all the data
    Iperm = randperm(length(xs))
    @time train!(n, adam, xs[Iperm], ys[Iperm], sumsquares)
end

# Plot real line and prediction
plot(-4:0.01:4, [fsol.(xi)[1] for xi in -4:0.01:4], c=:blue, legend =:false)
scatter!(xs, ys, lab="", m=(:cross,0.2,:blue),legend =:false)
scatter!(xs, [copy(n(xi)) for xi in xs], m=(:circle,0.2,:red), lab="",legend =:false)

# We can calculate the mean error over the training data like this also
getloss(n, xs, ys, sumsquares)
# Loss over test data like this
getloss(n, testxs, testys, sumsquares)
print("Mean error over training; ",string(getloss(n, xs, ys, sumsquares)),"\n","Mean error over test; ",string(getloss(n, testxs, testys, sumsquares)))

plot(-8:0.01:8, [fsol.(xi)[1] for xi in -8:0.01:8], c=:blue, label="Function",legend=:bottomright);
# Plot full network result
plot!(-8:0.01:8, [copy(n([xi]))[1] for xi in -8:0.01:8], c=:red, label="Network values", legend=:bottomright)
#########################################################
#########################################################
#########################################################
### Task 6:
fsol(x) = [min(0.5,sin(0.5*norm(x)^2))]
xs = [rand(2).*8 .- 4 for i = 1:2000] #30 for task 5 and 2000 for task 4 and 6
ys = [fsol(xi).+ 0.1.*randn(1) for xi in xs]

testxs = [rand(2).*8 .- 4 for i = 1:1000]
testys = [fsol(xi) for xi in testxs]

### Define network
# We use some reasonable value on initial weights
l1 = Dense(30, 2, leakyrelu, 0.0, 3.0, 0.0, 0.1)
lis = [Dense(30, 30, leakyrelu, 0.0, 3.0, 0.0, 0.1) for i = 1:4]
# Last layer has no activation function (identity)
ln = Dense(1, 30, identity, 0.0, 1.0, 0.0, 0.1)
n = Network([l1, lis..., ln])

### Define algorithm
adam = ADAMTrainer(n, 0.95, 0.999, 1e-8, 0.0001)

@time train!(n, adam, xs, ys, sumsquares)

# Train 100 times over the data set
for i = 1:100
    # Random ordering of all the data
    Iperm = randperm(length(xs))
    @time train!(n, adam, xs[Iperm], ys[Iperm], sumsquares)
end


getloss(n, xs, ys, sumsquares)
getloss(n, testxs, testys, sumsquares)
print("Mean error over training; ",string(getloss(n, xs, ys, sumsquares)),"\n","Mean error over test; ",string(getloss(n, testxs, testys, sumsquares)))

# Plotttnig that can be used for task 6:
scatter3d([xi[1] for xi in xs], [xi[2] for xi in xs], [n(xi)[1] for xi in xs], m=(:blue,1, :cross, stroke(0, 0.2, :blue)), size=(1200,800));
scatter3d!([xi[1] for xi in xs], [xi[2] for xi in xs], [yi[1] for yi in ys], m=(:red,1, :circle, stroke(0, 0.2, :red)), size=(1200,800))


### Task 6 new gamma:
adam = ADAMTrainer(n, 0.95, 0.999, 1e-8, 0.00001)

@time train!(n, adam, xs, ys, sumsquares)

# Train 100 times over the data set
for i = 1:100
    # Random ordering of all the data
    Iperm = randperm(length(xs))
    @time train!(n, adam, xs[Iperm], ys[Iperm], sumsquares)
end


getloss(n, xs, ys, sumsquares)
getloss(n, testxs, testys, sumsquares)
print("Mean error over training; ",string(getloss(n, xs, ys, sumsquares)),"\n","Mean error over test; ",string(getloss(n, testxs, testys, sumsquares)))


# Plottnig that can be used for task 6:
scatter3d([xi[1] for xi in xs], [xi[2] for xi in xs], [n(xi)[1] for xi in xs], m=(:blue,1, :cross, stroke(0, 0.2, :blue)), size=(1200,800));
scatter3d!([xi[1] for xi in xs], [xi[2] for xi in xs], [yi[1] for yi in ys], m=(:red,1, :circle, stroke(0, 0.2, :red)), size=(1200,800))

#########################################################
#########################################################
#########################################################

### Task 7
fsol(x) = [min(0.5,sin(0.5*norm(x)^2))]
xs = [rand(2).*8 .- 4 for i = 1:2000] #30 for task 5 and 2000 for task 4 and 6
ys = [fsol(xi).+ 0.1.*randn(1) for xi in xs]

testxs = [rand(2).*8 .- 4 for i = 1:1000]
testys = [fsol(xi) for xi in testxs]

### Define network
# We use some reasonable value on initial weights
l1 = Dense(30, 2, relu, 0.0, 3.0, 0.0, 0.1)
lis = [Dense(30, 30, relu, 0.0, 3.0, 0.0, 0.1) for i = 1:4]
# Last layer has no activation function (identity)
ln = Dense(1, 30, identity, 0.0, 1.0, 0.0, 0.1)
n = Network([l1, lis..., ln])

adam = ADAMTrainer(n, 0.95, 0.999, 1e-8, 0.00001)

@time train!(n, adam, xs, ys, sumsquares)

# Train 100 times over the data set
for i = 1:100
    # Random ordering of all the data
    Iperm = randperm(length(xs))
    @time train!(n, adam, xs[Iperm], ys[Iperm], sumsquares)
end


getloss(n, xs, ys, sumsquares)
getloss(n, testxs, testys, sumsquares)
print("Mean error over training; ",string(getloss(n, xs, ys, sumsquares)),"\n","Mean error over test; ",string(getloss(n, testxs, testys, sumsquares)))


# Plottnig that can be used for task 6:
scatter3d([xi[1] for xi in xs], [xi[2] for xi in xs], [n(xi)[1] for xi in xs], m=(:blue,1, :cross, stroke(0, 0.2, :blue)), size=(1200,800));
scatter3d!([xi[1] for xi in xs], [xi[2] for xi in xs], [yi[1] for yi in ys], m=(:red,1, :circle, stroke(0, 0.2, :red)), size=(1200,800))
