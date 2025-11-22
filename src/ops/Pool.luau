--!optimize 2
local Tensor = require("../Tensor")
local Tape = require("../autograd/Tape")
local Types = require("../Types")

-- MaxPool2d kernel KHxKW stride s; input {C,H,W,B}
--@param X Tensor
--@param KH number
--@param KW number
--@param stride number
--@return Tensor
return function(X: Types.Tensor, KH: number, KW: number, stride: number): Types.Tensor
	task.desynchronize()
	local C, H, W, B = X._shape[1], X._shape[2], X._shape[3], X._shape[4]
	local Ho = math.floor((H - KH) / stride) + 1
	local Wo = math.floor((W - KW) / stride) + 1
	local Y = Tensor.zeros({ C, Ho, Wo, B }, X._dtype, X._requiresGrad)
	local argmax = table.create(C * Ho * Wo * B, 0) -- store linear X indices
	for b = 1, B do
		for c = 1, C do
			for i = 1, Ho do
				for j = 1, Wo do
					local best = -1 / 0
					local bestXi = 0
					for ki = 1, KH do
						for kj = 1, KW do
							local ii = (i - 1) * stride + ki
							local jj = (j - 1) * stride + kj
							local xi = (((c - 1) * H + (ii - 1)) * W + (jj - 1)) * B + b - 1
							local v = X._storage[xi + 1]
							if v > best then
								best = v
								bestXi = xi
							end
						end
					end
					local yi = (((c - 1) * Ho + (i - 1)) * Wo + (j - 1)) * B + b
					Y._storage[yi] = best
					argmax[yi] = bestXi
				end
			end
		end
	end
	task.synchronize()
	return Tape.makeNode(Y, { X }, function(gy: Types.Tensor)
		if X._requiresGrad then
			X._grad = X._grad or Tensor.zeros(X._shape, X._dtype, false)
			local xGrad = X._grad :: Types.Tensor
			task.desynchronize()
			for idx = 1, #gy._storage do
				local xi = argmax[idx]
				if xi ~= 0 then
					xGrad._storage[xi + 1] += gy._storage[idx]
				end
			end
			task.synchronize()
		end
	end)
end
