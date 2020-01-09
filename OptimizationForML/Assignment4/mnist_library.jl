#MNIST-library

function create_A(K,class)
    A = zeros(K)
    A[class+1] = 1
    return A - (1/K).*ones(K)
end

function prox_grad_dual_coord(w,Q,g,stop)

    i = 0
    w_old = similar(w)
    tmp1 = similar(w)
    tmp1 .= w
    tmp2 = similar(w)
    tmp2 .= w

    n = length(w)

    x_iterates = Vector[]
    while i < stop
        i += 1
        random_index = rand(1:n)
        gamma = 1/Q[random_index,random_index]

        w_old .= tmp1
        tmp2[random_index] = Q[random_index,:]'*w_old
        tmp2[random_index] = w_old[random_index] - gamma*tmp2[random_index]
        tmp1[random_index] = prox(g, [tmp2[random_index]], gamma)[1][1]

        if mod(i,length(w)) == 0
            push!(x_iterates,copy(tmp1))
            print(string("\n","i= ",i," of ",stop))
        end
    end
    if i == stop
        print("\nProx_grad_dual_coord is finished: ")
        print(string("\nnorm:",norm(tmp1-w_old),"\ntime:"))
    end
    return tmp1,i,x_iterates
end

#Creating kernel function
function Kernel(x1,x2)
    return (dot(x1,x2))^5 
end
#-----------------------
#Creating Q
function make_Q(x_train,lambda,K,y_train)
    eye = Matrix{Float64}(I,K,K)
    dim = length(y_train)
    Q = zeros(dim,dim)

    for i = 1:dim
    Ai = create_A(K,y_train[i])
        for j = 1:dim
            Aj = create_A(K,y_train[j])
            Q[i,j] = (Kernel(x_train[i],x_train[j])/lambda)*Ai'*eye*Aj
        end
    end

    return Q
end

#-----------------------
#Classification model---

function classifier(x_hat,x_train,y_train,K,ny_sol,gamma)
    eye = Matrix{Float64}(I,K,K)
    dim = length(y_train)

    sum = zeros(K)
    for i = 1:dim
        A_i = create_A(K,y_train[i])
        sum +=  Kernel(x_hat,x_train[i])*eye*A_i*ny_sol[i]
    end

    Ci = zeros(K)
    #we check the confidence of each class, i.e we multiply with the different A:s that represent each class
    for class = 0:K-1
        Ai = copy(create_A(K,class)')
        Ci[class+1] = -gamma^-1*Ai*sum
    end

    return Ci
end

function errors(y_train,classes,x_train)
    tmp = 0
    indexer1 = 0
    indexer2 = 0
    wrongpix = []
    rightpix = []

    for i = 1:length(y_train)
        if  y_train[i] != classes[i]
            tmp+=1
        end
        if y_train[i] != classes[i] && indexer1 <=5
            push!(wrongpix,x_train[i])
            indexer1+=1
        end
        if y_train[i] == classes[i] && indexer2 <= 5
            push!(rightpix,x_train[i])
            indexer2+=1
        end
    end
    errorrate = tmp/length(y_train)
    return errorrate,wrongpix,rightpix
end
