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

#Generate data----------------------
Q,q,a,b = problem_data()
random = MersenneTwister(321) #Different starting x gives different solutions
#gamma = 2/opnorm(Q)
    #starting x---------------------
        x = randn(random,20)
    #-------------------------------

#Proximal gradient method-----------------------------------
function proximal_gradient(x,gamma,Q,q,a,b)
    snapshot = 0
    stop = 10^4
    x_old = 1000*x
    i = 1
    iter_to_conv = 0
    norm_diff = Float64[]
    #while norm(x_old-x) > 10^(-14) && i < stop
    while i < stop
        i += 1
        x_old = x
        differential_part = x_old-gamma*grad_quad(x_old,Q,q)
        x = prox_box(differential_part,a,b,gamma)
        #We want to save each 100:th value of the norm_diff
        if mod(i,100) == 0
            if norm(x_old-x) >= 1e-15
                append!(norm_diff,norm(x_old-x))
            else
                append!(norm_diff,1e-15)
            end
        end
        #------------------------------------------
        #We want to check how many iterations until the solution has converged
        if norm(x_old-x) <= 1e-15 && 0.9*1e-15 <= norm(x_old-x)
            iter_to_conv = i
        end
        #------------------------------------------
        #We want to take a snapshot of an iterate--
        if i == 100
            snapshot = x
        end
        #------------------------------------------
    end
    #append!(norm_diff,norm(x_old-x)) #adding the last value of the norm(x_old-x)
    return x,iter_to_conv,norm_diff,snapshot
end
#----------------------------------------------------------
#(test_solution, test_iterations, test_norm_diff, test_snap) = proximal_gradient(x,gamma,Q,q,a,b)

#testing different stepsizes------------------------------
iterations_until_conv = zeros(6)
norm_diff_for_plot = Vector{Float64}[]
primal_solutions = Vector{Float64}[]
snapshots = Vector{Float64}[]
for iter = 1:6
    gamma = (iter/5)*(2/opnorm(Q))
    (solution, nbr_of_it, norm_diff, snapshot) = proximal_gradient(x,gamma,Q,q,a,b)
    iterations_until_conv[iter] = nbr_of_it
    push!(primal_solutions,solution)
    push!(norm_diff_for_plot, norm_diff)
    push!(snapshots,snapshot)
    #print(string(gamma)," ", string(solution), " ", string(nbr_of_it), "\n")
end
#---------------------------------------------------------
#Plotting different norm_diffs
plt = plot(yscale =:log10, xlabel = "Iterations (x100)", ylabel = "||x_new-x_old||^2");
for i = 1:6
    plot!(plt,norm_diff_for_plot[i], label = string(i,"/5 x (2/L)"))
end
display(plt)

#--------------------------------------------------------
#printing the matrices-----------------------------------
reference_matrix_primevssnap = [round.(a,digits = 3) round.(primal_solutions[1],digits = 3) round.(snapshots[1],digits = 3) round.(b,digits = 3)]
reference_matrix_solvssol = [round.(a,digits = 3) round.(primal_solutions[1],digits = 3) round.(primal_solutions[2], digits = 3) round.(primal_solutions[6],digits = 3) round.(b,digits = 3)]
print(latexify(reference_matrix_solvssol))
print("\n")
print(latexify(reference_matrix_primevssnap))
#--------------------------------------------------------
