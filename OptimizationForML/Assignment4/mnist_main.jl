include("mnist_library.jl")
include("mnist.jl")

K = 10 #classes
red = 1 #reduction
(x_train, y_train) = loadmnist(0:K-1,reduction=red,set=:train)
(x_test,y_test) = loadmnist(0:K-1,reduction=red,set=:test)
N_test = length(x_test)
N = length(x_train)
lambda = 0.01
gamma = lambda

@time Q = make_Q(x_train,lambda,K,y_train)

my_start = randn(N)
y = ones(N)

f = HingeLoss([1],1/N)
f_conj = Conjugate(f)

@time ny_solution,iter,_ = prox_grad_dual_coord(my_start,Q,f_conj,500000)

confidences = Vector[]
@time for i = 1:N
    push!(confidences,classifier(x_train[i],x_train,y_train,K,ny_solution,gamma))
end

#making the classification:
classes = zeros(length(confidences))
for i = 1:length(confidences)
    (_,tmp1) = findmax(confidences[i])
    classes[i] = tmp1
end

classes = classes.-1

errorrate,wrongs,rights = errors(y_train,classes,x_train)

show_mnistimage(wrongs[1])
show_mnistimage(wrongs[2])
show_mnistimage(rights[1])
show_mnistimage(rights[2])

# now we examine a test set

confidences_test = Vector[]
@time for i = 1:N_test
    push!(confidences_test,classifier(x_test[i],x_train,y_train,K,ny_solution,gamma))
end

classes_test = zeros(length(confidences_test))
for i = 1:length(confidences_test)
    (_,tmp1) = findmax(confidences_test[i])
    classes_test[i] = tmp1
end
classes_test = classes_test.-1

errorrate_test,wrongs_test,rights_test = errors(y_test,classes_test,x_test)

show_mnistimage(wrongs_test[1])
show_mnistimage(wrongs_test[2])
show_mnistimage(rights_test[1])
show_mnistimage(rights_test[2])
