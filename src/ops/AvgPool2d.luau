--!optimize 2
local Tensor = require("../Tensor")
local Tape = require("../autograd/Tape")
local Types = require("../Types")

return function(X: Types.Tensor, KH: number, KW: number, stride: number): Types.Tensor
	task.desynchronize()
	local C, H, W, B = X._shape[1], X._shape[2], X._shape[3], X._shape[4]
	local Ho = math.floor((H - KH) / stride) + 1
	local Wo = math.floor((W - KW) / stride) + 1
	local Y = Tensor.zeros({ C, Ho, Wo, B }, X._dtype, X._requiresGrad)
	for b = 1, B do
		for c = 1, C do
			for i = 1, Ho do
				for j = 1, Wo do
					local sum = 0
					for ki = 1, KH do
						for kj = 1, KW do
							local ii = (i - 1) * stride + ki
							local jj = (j - 1) * stride + kj
							local xi = (((c - 1) * H + (ii - 1)) * W + (jj - 1)) * B + b - 1
							sum += X._storage[xi + 1]
						end
					end
					local yi = (((c - 1) * Ho + (i - 1)) * Wo + (j - 1)) * B + b
					Y._storage[yi] = sum / (KH * KW)
				end
			end
		end
	end
	task.synchronize()
	return Tape.makeNode(Y, { X }, function(gy: Types.Tensor)
		if X._requiresGrad then
			X._grad = X._grad or Tensor.zeros(X._shape, X._dtype, false)
			task.desynchronize()
			local scale = 1 / (KH * KW)
			local xGrad = X._grad :: Types.Tensor
			for b = 1, B do
				for c = 1, C do
					for i = 1, Ho do
						for j = 1, Wo do
							local yi = (((c - 1) * Ho + (i - 1)) * Wo + (j - 1)) * B + b
							local gyv = gy._storage[yi]
							for ki = 1, KH do
								for kj = 1, KW do
									local ii = (i - 1) * stride + ki
									local jj = (j - 1) * stride + kj
									local xi = (((c - 1) * H + (ii - 1)) * W + (jj - 1)) * B + b - 1
									xGrad._storage[xi + 1] += gyv * scale
								end
							end
						end
					end
				end
			end
			task.synchronize()
		end
	end)
end
