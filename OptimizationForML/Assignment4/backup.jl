#MNIST-library

function create_A_bkup(K,class)
    A1 = [1 ;0 ;0] - (1/K).*ones(K)
    A2 = [0 ;1 ;0] - (1/K).*ones(K)
    A3 = [0 ;0 ;1] - (1/K).*ones(K)
    if class == 0
        return A1
    elseif class == 1
        return A2
    elseif class == 2
        return A3
    end
end

function create_X(x_train,i)
    tmp = copy(x_train[i]')
    N = length(tmp)
    filler = copy(zeros(N)')
    X = [tmp filler filler;
        filler tmp filler;
        filler filler tmp]
    return copy(X')
end

function prox_grad_dual_coord(w,Q,g,stop) #nu kör vi den ett fixt antal gånger för att hitta en riktig lösning och sen en annan gång för att komma nära

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

        #tmp2 .= w_old.-tmp1

        if mod(i,length(w)) == 0
            push!(x_iterates,copy(tmp1))
            print(string("\n","i= ",i," of ",stop))
            #if norm(tmp2) < 1e-15
            #    break
            #end
        end
    end
    if i == stop
        print("\nProx_grad_dual_coord is finished: ")
    end
    return tmp1,i,x_iterates
end

#Creating kernel function
function Kernel(x1,x2)
    return (dot(x1,x2))^5 #/50 sen för att få allt samlat
end
#-----------------------
#Creating Q
function make_Q(x_train,lambda,K,y_train)
    eye = Matrix{Float64}(I,3,3)
    Q = zeros(6207,6207)

    for i = 1:6207
    Ai = create_A(K,y_train[i])
        for j = 1:6207
            Aj = create_A(K,y_train[j])
            Q[i,j] = (Kernel(x_train[i],x_train[j])/lambda)*Ai'*eye*Aj
        end
    end

    return Q
end

function model2(x_train,x_test,ny,Y,lambda)
    sum = 0
    term = copy(ny')*Y'
    for i = 1: length(x_train)
        sum +=term[i]*Kernel(x_train[i],x_test)
    end
    sum = sign(-sum)
    return sum
end
#-----------------------
#Classification model---

function classification(x_train,x_test,ny,Y,lambda)
    classification = zeros(length(x_test))
    for i = 1:length(x_test)
        classification[i] = model2(x_train,x_test[i],ny,Y,lambda)
    end
    return classification
end

function classifier(x_hat,x_train,y_train,K,ny_sol,gamma)
    eye = Matrix{Float64}(I,3,3)

    sum = zeros(3)
    for i = 1:6207
        A_i = create_A(K,y_train[i])
        sum +=  Kernel(x_hat,x_train[i])*eye*A_i*ny_sol[i]
    end

    Ci = zeros(3)
    for class = 0:2
        Ai = copy(create_A(K,class)')
        Ci[class+1] = -gamma^-1*Ai*sum
    end

    return Ci

end
