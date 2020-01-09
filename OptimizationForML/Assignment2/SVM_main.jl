using ProximalOperators
using LinearAlgebra
using Statistics
using Pkg
using Plots
using StatsPlots
Pkg.add("Plots")
include("problem2.jl")
include("functionlibrary3_new.jl")
gr()
(x_train,y_train) = svm_train()

Y = diagm(y_train)
N = length(x_train)
y = ones(N)
h = HingeLoss(y,inv(N))
h_conj = Conjugate(h)
sigma = 0.5
lambda = 1

Q = make_Q(x_train,y_train,lambda,sigma)
gamma = 1/opnorm(Q)

#Training variables ny-
ny_start = randn(N)
ny_solution,iter = prox_grad_dual(ny_start,Q,h_conj,gamma)
#----------------------

#Testing for task 6------
testervector = classification(x_train,x_train,ny_solution,Y,lambda,sigma)
error_test = errorcheck(testervector,y_train)
#------------------------

#Task 7------------------
lambdavector = [0.1 0.01 0.001 0.0001]
sigmavector = [1 0.5 0.25]
x_test,facit = svm_test_1()

error_rate_test = zeros(length(lambdavector),length(sigmavector))
error_rate_train = zeros(length(lambdavector),length(sigmavector))

for i = 1:length(lambdavector)
    for j = 1:length(sigmavector)
        lambda = lambdavector[i]
        sigma = sigmavector[j]
        Q = make_Q(x_train,y_train,lambda,sigma)
        gamma = 1/opnorm(Q)
        ny_solution,_ = prox_grad_dual(ny_start,Q,h_conj,gamma)
        classer_test = classification(x_train,x_test,ny_solution,Y,lambda,sigma)
        classer_train = classification(x_train,x_train,ny_solution,Y,lambda,sigma)
        error_rate_test[i,j] = errorcheck(classer_test,facit)
        error_rate_train[i,j] = errorcheck(classer_train,y_train)
    end
end
besterr, i, j = besterror(error_rate_test)
print(string("Best error rate was ",string(besterr), " with λ being ",string(lambdavector[i]), " and σ being ",string(sigma)))

best_lambda = lambdavector[i]
best_sigma = sigmavector[j]

#Task 8----------------
lambda = best_lambda
sigma = best_sigma
Q = make_Q(x_train,y_train,lambda,sigma)
gamma = 1/opnorm(Q)
ny_solution,_ = prox_grad_dual(ny_start,Q,h_conj,gamma)

error_rate_vector = zeros(4)

x_test_1,facit_1 = svm_test_1()
x_test_2,facit_2 = svm_test_2()
x_test_3,facit_3 = svm_test_3()
x_test_4,facit_4 = svm_test_4()

x_tests = Vector[]
push!(x_tests,x_test_1)
push!(x_tests,x_test_2)
push!(x_tests,x_test_3)
push!(x_tests,x_test_4)

facits = Vector[]
push!(facits,facit_1)
push!(facits,facit_2)
push!(facits,facit_3)
push!(facits,facit_4)

for i = 1:4
    classer_test = classification(x_train,x_tests[i],ny_solution,Y,lambda,sigma)
    error_rate_vector[i] = errorcheck(classer_test,facits[i])
end

print("We get the following error rates for each data set: ",string(error_rate_vector))

#Task 8 part 2 - we want to check each given pair of lambda and sigma for all val-sets
lambda_sigma_pairs = [0.1 0.001 0.00001; 2 0.5 0.25]
all_error_rates_test = Array[]
all_error_rates_train = Array[]

for k = 1:4
    error_rate_test = zeros(3)
    error_rate_train = zeros(3)
    for i = 1:length(lambda_sigma_pairs[1,:])
            lambda = lambda_sigma_pairs[1,i]
            sigma = lambda_sigma_pairs[2,i]
            Q = make_Q(x_train,y_train,lambda,sigma)
            gamma = 1/opnorm(Q)
            ny_solution,_ = prox_grad_dual(ny_start,Q,h_conj,gamma)
            classer_test = classification(x_train,x_tests[k],ny_solution,Y,lambda,sigma)
            classer_train = classification(x_train,x_train,ny_solution,Y,lambda,sigma)
            error_rate_test[i] = errorcheck(classer_test,facits[k])
            error_rate_train[i] = errorcheck(classer_train,y_train)
    end
    push!(all_error_rates_test,error_rate_test)
    push!(all_error_rates_train,error_rate_train)
end

#Task 8 part 3 - we want to check for different hyperparameters

error_rates_test = Array[]
error_rates_train = Array[]

for k = 1:4
    error_rate_test = zeros(4,3)
    error_rate_train = zeros(4,3)
    for i = 1:length(lambdavector)
        for j = 1:length(sigmavector)
            lambda = lambdavector[i]
            sigma = sigmavector[j]
            Q = make_Q(x_train,y_train,lambda,sigma)
            gamma = 1/opnorm(Q)
            ny_solution,_ = prox_grad_dual(ny_start,Q,h_conj,gamma)
            classer_test = classification(x_train,x_tests[k],ny_solution,Y,lambda,sigma)
            classer_train = classification(x_train,x_train,ny_solution,Y,lambda,sigma)
            error_rate_test[i,j] = errorcheck(classer_test,facits[k])
            error_rate_train[i,j] = errorcheck(classer_train,y_train)
        end
    end
    push!(error_rates_test,error_rate_test)
end

#Task 9---10-fold-------------
#lambda = best_lambda
#sigma = best_sigma
function assign9a()
    lambda = 0.0001
    sigma = 0.5

    data,facit = svm_train()
    new_data, new_facit = scramble(data,facit)

    valfacit_fold = similar(new_facit,50)
    val_fold = similar(new_data,50)
    facit_fold = similar(new_facit,450)
    data_fold = similar(new_data,450)
    ny_start = randn(450)
    y = ones(450)
    h = HingeLoss(y,450)
    h_conj = Conjugate(h)
    histnr = 100
    error_rate = zeros(histnr)
    #error_rate_test = zeros(10)
    for k = 1:histnr
        new_data, new_facit = scramble(data,facit)
        for i = 1:10
            getFoldedData!(new_data,new_facit,valfacit_fold,val_fold,facit_fold,data_fold,i,10,50)
            Y = diagm(facit_fold)
            Q = make_Q(data_fold,facit_fold,lambda,sigma)
            gamma = 1/opnorm(Q)
            ny_solution,_ = prox_grad_dual(ny_start,Q,h_conj,gamma)
            classer_test = classification(data_fold,val_fold,ny_solution,Y,lambda,sigma)
            error_rate[k] += errorcheck(classer_test,valfacit_fold)
            print(string("\n",errorcheck(classer_test,valfacit_fold)))
        end
        error_rate[k] /=10
        print(string("\nAverage error rate: ",string(error_rate[k])))
        print(string("\nk: ",k))
    end
    global plt_obja = plot(xlabel = "Error rate", ylabel = "Observations")
    histogram!(plt_obja,error_rate, bins= convert(Int64,round(histnr/4, digits = 0)))
    #display(p)
    return error_rate
end
plt_obja = plot()
error_rate = assign9a()
display(plt_obja)

function assign9b()
    lambda = 0.0001
    sigma = 0.5

    data,facit = svm_train()
    new_data, new_facit = scramble(data,facit)

    valfacit_fold = similar(new_facit,100)
    val_fold = similar(new_data,100)
    facit_fold = similar(new_facit,400)
    data_fold = similar(new_data,400)
    ny_start = randn(400)
    y = ones(400)
    h = HingeLoss(y,400)
    h_conj = Conjugate(h)
    histnr = 100
    error_rate = zeros(histnr)
    #error_rate_test = zeros(10)
    for k = 1:histnr
        new_data, new_facit = scramble(data,facit)
        getFoldedData!(new_data,new_facit,valfacit_fold,val_fold,facit_fold,data_fold,1,5,100)
        Y = diagm(facit_fold)
        Q = make_Q(data_fold,facit_fold,lambda,sigma)
        gamma = 1/opnorm(Q)
        ny_solution,_ = prox_grad_dual(ny_start,Q,h_conj,gamma)
        classer_test = classification(data_fold,val_fold,ny_solution,Y,lambda,sigma)
        error_rate[k] = errorcheck(classer_test,valfacit_fold)
        print(string("\n",errorcheck(classer_test,valfacit_fold)))
        print(string("\nk: ",k))
    end
    global plt_objb = plot(xlabel = "Error rate", ylabel = "Observations")
    histogram!(plt_objb,error_rate, bins= convert(Int64,histnr/4))
end

assign9b()
