using MLDatasets
using Plots
using ProximalOperators
using LinearAlgebra
using Statistics
using Pkg
using StatsPlots
using SparseArrays

"""
	show_mnistimage(img)

Displays one image loaded with the loadmnist function.
"""
function show_mnistimage(img)
	plot(Gray.(reshape(img,28,28)))
end

"""
	x,y = loadmnist(labels; set=:train, reduction=1)

# Arguments:
- labels: set of labels to load, e.g. [0,1] or 0:4
- set: training or test data, i.e. :train or :test
- reduction: a reduction factor, i.e. load roughly 1/<reduction> of the images

"""
function loadmnist(labels;set=:train,reduction=1)
	if set == :train
		x_raw, y_raw = MNIST.traindata()
	elseif set == :test
		x_raw, y_raw = MNIST.testdata()
	end
	n_col,n_row,n_img = size(x_raw)

	function filter_reduce!(selection, label)
		count = 1
		for (i, yi) in enumerate(y_raw)
			yi != label && continue
			count += 1
			(count % reduction != 0) && continue
			selection[i] = true
		end
	end

	selection = zeros(Bool,n_img)
	foreach(lab -> filter_reduce!(selection,lab), labels)
	n_selection = count(selection)
	idx_selection = (1:n_img)[selection]

	x_eltype = Float64
	y_type = Int

	x = Vector{Vector{x_eltype}}(undef, n_selection)
	y = Vector{y_type}(undef, n_selection)

	for (i, i_sel) in enumerate(idx_selection)
		y[i] = y_type(y_raw[i_sel])

		# Saturate and vectorize
		x[i] = reshape(x_eltype.(round.(view(x_raw, :,:,i_sel)')),:)
	end

	return x,y
end
