"""
    quad(x,Q,q)

Compute the quadratic

	1/2 x'Qx + q'x

"""
function quad(x,Q,q)
	return 1/2 * x'*Q*x + q'*x
end



"""
    guadconj(y,Q,q)

Compute the convex conjugate of the quadratic

	1/2 x'Qx + q'x

"""
function quadconj(y,Q,q)
	return 1/2 * (y-q)'*inv(Q)*(y-q)
end



"""
    box(x,a,b)

Compute the indicator function of for the box contraint

	a <= x <= b

where the inequalites are applied element-wise.
"""
function box(x,a,b)
	return all(a .<= x .<= b) ? 0.0 : Inf
end



"""
    boxconj(y,a,b)

Compute the convex conjugate of the indicator function of for the box contraint

	a <= x <= b

where the inequalites are applied element-wise.
"""
function boxconj(y,a,b)
	sum = 0
	for i = 1:length(y)
		biggest = maximum(y[i]*a[i],y[i]*b[i])
		sum = sum + biggest
	end
	return sum
end





"""
    grad_quad(x,Q,q)

Compute the gradient of the quadratic

	1/2 x'Qx + q'x

"""
function grad_quad(x,Q,q)
	return Q*x + q
end



"""
    grad_quadconj(y,Q,q)

Compute the gradient of the convex conjugate of the quadratic

	1/2 x'Qx + q'x

"""
function grad_quadconj(y,Q,q)
	return inv(Q)*(y-q)
end



"""
    prox_box(x,a,b)

Compute the proximal operator of the indicator function for the box contraint

	a <= x <= b

where the inequalites are applied element-wise.
"""

function prox_box(x,a,b,gamma)
	ret = []
	for i in 1:length(x)
		if x[i] < a[i]
			append!(ret,a[i])
		elseif a[i] <= x[i] <= b[i]
			append!(ret,x[i])
		elseif b[i] < x[i]
			append!(ret,b[i])
		end
	end
	return ret
end


"""
    prox_boxconj(y,a,b)

Compute the proximal operator of the convex conjugate of the indicator function
for the box contraint

	a <= x <= b

where the inequalites are applied element-wise.
"""
function prox_boxconj(y,a,b,gamma)
	temp = zeros(size(y))
		for i = 1:length(y)
			if (y[i]/gamma) < a[i]
				temp[i] = y[i] - gamma*a[i]
			elseif (y[i]/gamma) > b[i]
				temp[i] = y[i] - gamma*b[i]
			else
				temp[i] = 0 #y[i] - gamma*y[i]
			end
		end
	return temp
end


"""
    dual2primal(y,Q,q,a,b)

Computes the solution to the primal problem for Hand-In 1 given a solution y to
the dual problem.
"""
function dual2primal(y,Q,q,a,b)
	return inv(Q)*(y - q)
end
