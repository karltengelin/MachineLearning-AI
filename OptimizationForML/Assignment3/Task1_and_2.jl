using ProximalOperators
using LinearAlgebra
using Statistics
using Pkg
using Plots
using StatsPlots
Pkg.add("Plotly")
Pkg.add("Plots")
include("problem_from_2.jl")
include("functionlibrary3_new.jl")
#gr()
plotly()

function task1()
    random = MersenneTwister(31)
    (x_train,y_train) = svm_train()

    Y = diagm(y_train)
    N = length(x_train)
    y = ones(N)
    h = HingeLoss(y,inv(N))
    h_conj = Conjugate(h)
    sigma = 0.5
    lambda = 0.0001

    Q = make_Q(x_train,y_train,lambda,sigma)
    gamma = 1/opnorm(Q)

    #Training variables ny-
    ny_start = randn(random,N)
    global ny_solution,iter,x_iterates = prox_grad_dual_acc2(ny_start,Q,h_conj,gamma,99,1)
    global ny_solution2,iter2,x_iterates2 = prox_grad_dual_acc2(ny_start,Q,h_conj,gamma,99,2)
    #training without acceleration
    global ny_solution_old,iter_old,x_iterates_old = prox_grad_dual(ny_start,Q,h_conj,gamma)
    #print(string("\niter1: ",iter))
    #print(string("\niter2: ",iter2))
    #----------------------

    #Testing for task 1------
    testervector = classification(x_train,x_train,ny_solution,Y,lambda,sigma)
    error_test = errorcheck(testervector,y_train)
    testervector2 = classification(x_train,x_train,ny_solution2,Y,lambda,sigma)
    error_test2 = errorcheck(testervector2,y_train)
    testervector_old = classification(x_train,x_train,ny_solution_old,Y,lambda,sigma)
    error_test_old = errorcheck(testervector_old,y_train)
    #print(ny_solution)
    global plotthing = zeros(iter-1)
    for i = 1:iter-1
        plotthing[i] = norm(x_iterates[i]-ny_solution)
    end
    global plotthing2 = zeros(iter2-1)
    for i = 1:iter2-1
        plotthing2[i] = norm(x_iterates2[i]-ny_solution2)
    end
    global plotthing_old = zeros(iter_old-1)
    for i = 1:iter_old-1
        plotthing_old[i] = norm(x_iterates_old[i]-ny_solution_old)
    end
    p1 = plot(plotthing, yaxis=:log10,xlabel = "iterations", ylabel = "errornorm")
    p2 = plot(plotthing2, yaxis=:log10,xlabel = "iterations", ylabel = "errornorm")
    p_old = plot(plotthing_old, yaxis=:log10,xlabel = "iterations", ylabel = "errornorm")
    display(p1)
    display(p2)
    display(p_old)
    print(string("\nerror: ", error_test))
    print(string("\niterations: ", iter))
    print(string("\nerror2: ", error_test2))
    print(string("\niterations2: ", iter2))
    print(string("\nerror_old: ", error_test_old))
    print(string("\niterations_old: ", iter_old))
end
task1()

#task2
#include("functionlibrary3_new.jl")
function task2()
    random = MersenneTwister(31)
    (x_train,y_train) = svm_train()

    Y = diagm(y_train)
    N = length(x_train)
    y = ones(N)
    #h = HingeLoss([1],1.0)
    #This surprisingly gives goodish results for gamma is 1/opnorm: h = HingeLoss([1],1)
    sigma = 0.5
    lambda = 0.0001

    Q = make_Q(x_train,y_train,lambda,sigma)
    gamma = 1/opnorm(Q)

    h = HingeLoss(y,inv(N))
    h_conj = Conjugate(h)

    ny_start = randn(random,N)
    @time ny_solution_old,iter_old,x_iterates_old = prox_grad_dual(ny_start,Q,h_conj,gamma)
    #Training variables ny
    h = HingeLoss([1],1/N)
    #h = HingeLoss(y,inv(N))
    h_conj = Conjugate(h)

    @time ny_solution,iter,x_iterates = prox_grad_dual_coord(ny_start,Q,h_conj,gamma,35000000,ny_solution_old)
    #global _,iter,x_iterates = prox_grad_dual_coord(ny_start,Q,h_conj,gamma,10000000)
    #print(string("iter1: ",iter))

    #Testing for task 2------
    testervector = classification(x_train,x_train,ny_solution,Y,lambda,sigma)
    error_test = errorcheck(testervector,y_train)
    print(string("\nerror: ", error_test))
    print(string("\niterations: ", iter))

    plotthing = zeros(length(x_iterates))
    for i = 1:length(x_iterates)
        plotthing[i] = norm(x_iterates[i]-ny_solution_old)
    end

    return plotthing
end

plotthing = task2()

p1 = plot(plotthing[1:end], yaxis=:log10, ylabel = "errornorm", xlabel = "1000x iterates")

display(p1)
