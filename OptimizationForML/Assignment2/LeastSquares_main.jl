using ProximalOperators
using LinearAlgebra
using Statistics
using Pkg
using Plots
using StatsPlots
using Latexify
#using DataFrames
Pkg.add("Plots")
Pkg.add("StatsPlots")
include("problem2.jl")
include("functionlibrary3_new.jl")
#gr()
plotly()

#Rescaling
function rescale(x)
    new_x = zeros(length(x),1)
    beta = 0
    sigma = 1/maximum(abs.(x))
        for i = 1:length(x)
            new_x[i] = (x[i]-beta)*sigma
        end
    return new_x
end

#Generate data
function generate_data(p)
    (x,y) = leastsquares_data()
    r = rescale(x) #x #rescaled x-values
    rand = MersenneTwister(321)
    w = randn(rand,p+1)
    return w,r
end

#Starting the gradient approach-----

function prox_grad(w, X, y, gamma,g)
    f = LeastSquares(copy(X'),y,1.0)
    stop =  100000000 #20000000
    w_old = 1000*w
    i = 0
        while norm(w_old-w) > 10^(-15) && i < stop
            i += 1
            w_old = w
            grad,_ = gradient(f,w_old)       #X'w_old-y
            w,_ = prox(g, w_old-gamma*grad, gamma)
        end
    norm_val = norm(w_old-w)
    return w,i,norm_val
end

#--------------------------------------
#assign1
#w,psi,X,r,y = generate_data(1)
#gamma = inv(norm(X'X))
#w,i = prox_grad(w, X, y, gamma)

#plt = plot(psi*w, label = "model")
#plot!(plt,r, label = "r-values")
#print("Resulting point: ")
#print(w)
#print("\nNr of iterations: ")
#print(i)
#--------------------------------------

function psi(x,p)
    out = [x^i for i in 0:p]
    return out
end

function create_X(x,p)
    X = zeros(p+1,length(x))
    for j = 1:length(x)
        X[:,j] = psi(x[j],p)
    end
    return X
end

#Task 2
(x,y) = leastsquares_data()
r = rescale(x)
g = NormL2(0.0)
plt = scatter(x,y,legend=:topright, label = "y-values", xlabel="x", ylabel= "m(r(x))")
#scatter!(plt,r, label = "r-values")
for p = 1:10
    w,r= generate_data(p)
    X = create_X(r,p)
    gamma = inv(opnorm(X'X))
    w_new,itr = prox_grad(w, X, y, gamma, g)
    print(w_new)

    m(x,p) = w_new'*psi(x,p)
    plotstuff = m.(rescale(-1.1:0.01:3.10),p)
    plot!(plt,-1.1:0.01:3.10,plotstuff,label = string("p ="," ",string(p)))
    print("\nNr of iterations of p = ")
    print(p)
    print(": ")
    print(itr)
end
display(plt)

#------------------------------------

#Task 3------------------------------
p = 10
(x,y) = leastsquares_data()
w,r = generate_data(p)
X = create_X(r,p)
gamma = inv(opnorm(X'X))
plt_model = scatter(x,y,legend = :topright, xlabel = "x", ylabel="m(r(x))")
#plt4 = plot(legend = :topright)
lambdavec = [0.001 0.01 0.1 1 10]        #0.001:0.9:10
iterations = zeros(length(lambdavec))
w_star = Array[]

for iter = 1:length(lambdavec)
    lambda = lambdavec[iter]
    g = SqrNormL2(lambda)
    #g = NormL1(lambda)
    w_new,itr = prox_grad(w, X, y, gamma, g)
    iterations[iter] = itr
    push!(w_star,w_new)

    m(x,p) = w_new'*psi(x,p)
    plotstuff = m.(rescale(-1.1:0.01:3.10),p)
    plot!(plt_model,-1.1:0.01:3.10,plotstuff,label = string("λ ="," ",string(lambda)))
    print("\nNr of iterations of λ = ")
    print(lambda)
    print(": ")
    print(itr)

    #model = psi*w_new
    #plot!(plt3,model, label = string("lambda"," ",string(lambda)))
    #scatter!(plt4, w_new)
end

display(plt_model)

#Task 4------------------------------
p = 10
(x,y) = leastsquares_data()
w,_ = generate_data(p)
r=x
X = create_X(r,p)
gamma = inv(opnorm(X'X))
plt_3 = scatter(x,y,legend = :topright, xlabel = "x", ylabel="m(x)")
lambda = 0.01
#g = SqrNormL2(lambda) #we choose q = 2 so the 2-norm
g = NormL1(lambda)
w_new,itr,norm_val = prox_grad(w, X, y, gamma, g)
m(x,p) = w_new'*psi(x,p)
plotstuff = m.(rescale(-1.1:0.01:3.10),p)
plot!(plt_3,-1.1:0.01:3.10,plotstuff,label = string("λ ="," ",string(lambda)))


#display(plt4)
