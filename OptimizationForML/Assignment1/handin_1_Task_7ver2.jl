using LinearAlgebra
using Pkg
Pkg.add("Plots")
Pkg.add("Latexify")
using Latexify
using Plots
include("problem.jl")
include("functions.jl")
plotly()
gr()
#using Gadfly

#Generate data----------------------
Q,q,a,b = problem_data()
random = MersenneTwister(321) #Different starting x gives different solutions
#gamma = 2/opnorm(Q)
    #starting x---------------------
        x = randn(random,20)
    #-------------------------------

#Proximal gradient method-----------------------------------
function proximal_gradient_dual(x,gamma,Q,q,a,b)
    snapshot = 0
    stop = 150000
    x_old = 1000*x
    i = 1
    iter_to_conv = 0
    norm_diff = Float64[]
        #while norm(x_old-x) > 10^(-14) && i < stop
    itersave = false
    function_values = Float64[]
    while i < stop
        i += 1
        x_old = x
        differential_part = x_old-gamma*grad_quadconj(x_old,Q,q)
        x = -prox_boxconj(-differential_part,a,b,gamma)                 #prox_boxconj(differential_part,a,b,gamma)
        #We want to save each 100:th value of the norm_diff
        if mod(i,100) == 0
            #print(quad(dual2primal(x,Q,q,a,b),Q,q))
            append!(function_values, quad(dual2primal(x,Q,q,a,b),Q,q))
            if norm(x_old-x) >= 1e-15
                append!(norm_diff,norm(x_old-x))
            else
                append!(norm_diff,1e-15)
                #iter_to_conv = i
            end
        end
        if norm(x_old-x) < 1e-15 && itersave == false
            itersave = true
            iter_to_conv = i
        end
        #------------------------------------------
        #We want to take a snapshot of an iterate--
        if i == 100
            snapshot = x
        end
        #if i == stop
        #    print("\niterationtest: ",iter_to_conv)
        #end
        #------------------------------------------
    end
    #append!(norm_diff,norm(x_old-x)) #adding the last value of the norm(x_old-x)
    return x,iter_to_conv,norm_diff,snapshot,function_values
end
#----------------------------------------------------------
#(test_solution, test_iterations, test_norm_diff) = proximal_gradient(x,gamma,Q,q,a,b)

#testing different stepsizes------------------------------
iterations_until_conv = zeros(5)
norm_diff_for_plot = Vector{Float64}[]
dual_solutions = Vector{Float64}[]
snapshots = Vector{Float64}[]
func_val = Vector{Float64}[]
for iter = 1:5
    gamma = (iter/5)*(2/opnorm(inv(Q)))
    (solution, nbr_of_it, norm_diff, snapshot,func_values) = proximal_gradient_dual(x,gamma,Q,q,a,b)
    iterations_until_conv[iter] = nbr_of_it
    push!(dual_solutions,solution)
    push!(norm_diff_for_plot, norm_diff)
    push!(snapshots,snapshot)
    push!(func_val,func_values)
    #print(string(gamma)," ", string(solution), " ", string(nbr_of_it), "\n")
end
#---------------------------------------------------------
#Plotting different norm_diffs
plt = plot(yscale =:log10, xlabel = "Iterations (x100)", ylabel = "||x_new-x_old||^2");
for i = 1:5
    plot!(plt,norm_diff_for_plot[i], label = string(i,"/5 x (2/L*)"))
    print("\nIterations until convergence: ",iterations_until_conv[i],"\n")
end
plt2 = plot(xlabel = "Iterations (x100)", ylabel = "f(x)");
plot!(plt2,func_val[5])
display(plt)
display(plt2)

dual2primal_solutions = Vector{Float64}[]
for i = 1:5
    push!(dual2primal_solutions,dual2primal(dual_solutions[i],Q,q,a,b))
end

#printing the matrices-----------------------------------
reference_matrix_dual2primevsdualsnap = [round.(a,digits = 3) round.(snapshots[5],digits = 3) round.(b,digits = 3)]
reference_matrix_solvssol_dual = [round.(a,digits = 13) round.(dual2primal_solutions[5],digits = 14) round.(b,digits = 14)]
#reference_matrix_dualandprimal = [round.(primal_solutions[5],digits = 14) round.(dual2primal_solutions[5],digits = 14)]
print(latexify(reference_matrix_solvssol_dual))
print("\n")
print(latexify(reference_matrix_dual2primevsdualsnap))
#print(latexify(reference_matrix_dualandprimal))
#--------------------------------------------------------
