--!optimize 2
local Tensor = require("../Tensor")
local Tape = require("../autograd/Tape")

return function(V, A)
	local ADim, B = A._shape[1], A._shape[2]
	local Q = Tensor.zeros({ ADim, B }, A._dtype, V._requiresGrad or A._requiresGrad)
	task.desynchronize()
	for j = 1, B do
		local meanA = 0
		for i = 1, ADim do
			meanA += A._storage[(i - 1) * B + j]
		end
		meanA /= ADim
		for i = 1, ADim do
			local idx = (i - 1) * B + j
			Q._storage[idx] = V._storage[j] + A._storage[idx] - meanA
		end
	end
	task.synchronize()
	return Tape.makeNode(Q, { V, A }, function(gQ)
		if V._requiresGrad then
			if not V._grad then
				V._grad = Tensor.zeros({ 1, B })
			end
			task.desynchronize()
			for j = 1, B do
				local acc = 0
				for i = 1, ADim do
					acc += gQ._storage[(i - 1) * B + j]
				end
				V._grad._storage[j] += acc
			end
			task.synchronize()
		end
		if A._requiresGrad then
			if not A._grad then
				A._grad = Tensor.zeros({ ADim, B })
			end
			task.desynchronize()
			for j = 1, B do
				local acc = 0
				for i = 1, ADim do
					acc += gQ._storage[(i - 1) * B + j]
				end
				local meanGrad = acc / ADim
				for i = 1, ADim do
					local idx = (i - 1) * B + j
					A._grad._storage[idx] += gQ._storage[idx] - meanGrad
				end
			end
			task.synchronize()
		end
	end)
end
