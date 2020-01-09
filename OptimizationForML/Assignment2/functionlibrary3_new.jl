function prox_grad_dual_acc2(w,Q,g,gamma,mu,betaswitch)
    #shorter version
    f = Quadratic(Q,zeros(size(Q)[1]))
    stop = 1000000
    i = 0
    w_old = similar(w)
    tmp1 = similar(w)
    tmp1 .= w
    tmp2 = similar(w)
    tmp2 .= w
    whalf = similar(w)
    told = 1
    x_iterates = Vector[]

    while norm(tmp2) > 10^(-10) && i < stop
        if betaswitch == 1
            beta = (i-2)/(i+1)
        elseif betaswitch == 2
            tnew = (1 + sqrt((1+4*told^2)))/2
            beta = (told-1)/tnew
            told = tnew
        elseif betaswitch == 3
            beta = (1-sqrt(mu*gamma))/(1+sqrt(mu*gamma))
        end
        i += 1

        whalf .= w_old
        w_old .= tmp1
        whalf .= w_old .+ beta .* (w_old .- whalf)
        gradient!(tmp2,f,whalf)
        tmp2 .= whalf .- gamma.*tmp2
        prox!(tmp1,g, tmp2 , gamma)
        tmp2.= w_old.-tmp1
        push!(x_iterates,copy(tmp1))
    end
    if i == stop
        print("\nProx_grad_dual did not converge :(")
    end
    return tmp1,i,x_iterates
end

function prox_grad_dual_coord(w,Q,g,gamma,stop) #nu kör vi den ett fixt antal gånger för att hitta en riktig lösning och sen en annan gång för att komma nära

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
        #activate for adapted gamma
        gamma = 1/Q[random_index,random_index]

        #this should be correct, I think?
        w_old .= tmp1
        tmp2[random_index] = Q[random_index,:]'*w_old
        #the same: tmp2[random_index] = (Q*w_old)[random_index]
        tmp2[random_index] = w_old[random_index] - gamma*tmp2[random_index]
        tmp1[random_index] = prox(g, [tmp2[random_index]], gamma)[1][1]
        tmp2 .= w_old.-tmp1

        if mod(i,length(w)) == 0
            push!(x_iterates,copy(tmp1))
            #if norm(tmp2) < 1e-15
            #    break
            #end
        end
    end
    if i == stop
        print("\nProx_grad_dual did not converge 😞")
    end
    return tmp1,i,x_iterates
end

function prox_grad_dual(w,Q,g,gamma)
    f = Quadratic(Q,zeros(size(Q)[1]))
    stop = 1000000
    i = 0
    w_old = similar(w)
    tmp1 = similar(w)
    tmp1 .= w
    tmp2 = similar(w)
    tmp2 .= w
    x_iterates = Vector[]

    while norm(tmp2) > 10^(-10) && i < stop
        i += 1
        w_old .= tmp1
        gradient!(tmp2,f,w_old)
        tmp2 .= w_old .- gamma.*tmp2
        prox!(tmp1,g, tmp2 , gamma)
        tmp2.= w_old.-tmp1
        push!(x_iterates,copy(tmp1))
    end
    if i == stop
        print("\nProx_grad_dual did not converge :(")
    end
    return tmp1,i,x_iterates
end

#Creating kernel function
function Kernel(x1,x2,sigma)
    temp = exp((-1/(2*sigma^2))*(norm(x1-x2))^2)
    return temp
end
#-----------------------
#Creating Q
function make_Q(x,y,lambda,sigma)
    dim1 = length(x)
    dim2 = length(y)
    Q = zeros(dim1,dim2)
    for i = 1:dim1
        for j = 1:dim2
            Q[i,j] = (1/lambda)*y[i]*Kernel(x[i],x[j],sigma)*y[j]
        end
    end
    return Q
end
#------------------------
#Our model
function model(x_train,x_test,ny,Y,lambda,sigma)
    vector = zeros(length(x_train))
    for i = 1:length(x_train)
        vector[i] = Kernel(x_train[i],x_test,sigma)
    end
    temp = sign((-1/lambda)*copy(ny')*copy(Y')*vector)
    return temp
end

function model2(x_train,x_test,ny,Y,lambda,sigma)
    sum = 0
    term = copy(ny')*Y'
    for i = 1: length(x_train)
        sum +=term[i]*Kernel(x_train[i],x_test,sigma)
    end
    sum = sign(-sum)
    return sum
end
#-----------------------
#Classification model---

function classification(x_train,x_test,ny,Y,lambda,sigma)
    classification = zeros(length(x_test))
    for i = 1:length(x_test)
        classification[i] = model2(x_train,x_test[i],ny,Y,lambda,sigma)
    end
    return classification
end

#error check-----------
function errorcheck(classification,facit)
    temp = 0
    iter = length(classification)
    for i = 1:iter
        if classification[i] != facit[i]
            temp += 1
        end
    end
    return temp/iter
end
#---------------------

#best error rate
function besterror(errorrate)
    #returns lowest error rate with respective indices of lambda and sigma
    err = Inf
    best_i = 0
    best_j = 0
    i_i = length(errorrate[:,1])
    j_j = length(errorrate[1,:])
    for i = 1:i_i
        for j = 1:j_j
            if errorrate[i,j] < err
                err = errorrate[i,j]
                best_i = i
                best_j = j
            end
        end
    end
    return err,best_i,best_j
end

function scramble(data,facit)
    indexvector = randperm(length(data))
    new_data = data[indexvector]
    new_facit = facit[indexvector]
    return new_data, new_facit
end

function getFoldedData!(new_data,new_facit,valfacit_fold,val_fold,facit_fold,data_fold,i,nris,fold_size)
    valfacit_fold .= new_facit[(i-1)*fold_size+1:i*fold_size]
    val_fold .= new_data[(i-1)*fold_size+1:i*fold_size]

    for j = 1:nris-1
        if j < i
            facit_fold[(j-1)*fold_size+1:j*fold_size] .= new_facit[(j-1)*fold_size+1:j*fold_size]
            data_fold[(j-1)*fold_size+1:j*fold_size]  .= new_data[(j-1)*fold_size+1:j*fold_size]
        elseif j >= i
            facit_fold[(j-1)*fold_size+1:j*fold_size] .= new_facit[(j)*fold_size+1:(j+1)*fold_size]
            data_fold[(j-1)*fold_size+1:j*fold_size]  .= new_data[(j)*fold_size+1:(j+1)*fold_size]
        end
    end
    return nothing
end
