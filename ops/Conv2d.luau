--!optimize 2
local Tensor = require("../Tensor")
local Tape = require("../autograd/Tape")
local Types = require("../Types")

--@param X Tensor
--@param W Tensor -- {F,C,KH,KW}
--@return Tensor -- {F,Ho,Wo,B}
return function(X: Types.Tensor, W: Types.Tensor): Types.Tensor
	local C, H, Wd, B = X._shape[1], X._shape[2], X._shape[3], X._shape[4]
	local F, Cw, KH, KW = W._shape[1], W._shape[2], W._shape[3], W._shape[4]
	assert(C == Cw, "conv2d: channel mismatch")
	task.desynchronize()
	local Ho = (H - KH) + 1
	local Wo = (Wd - KW) + 1
	local Y = Tensor.zeros({ F, Ho, Wo, B }, X._dtype, X._requiresGrad or W._requiresGrad)
	local xs, ws, ys = X._storage, W._storage, Y._storage
	for b = 1, B do
		for f = 1, F do
			for i = 1, Ho do
				for j = 1, Wo do
					local acc = 0
					for c = 1, C do
						for ki = 1, KH do
							for kj = 1, KW do
								local xi = (((c - 1) * H + (i + ki - 2)) * Wd + (j + kj - 2)) * B + b
								local wi = ((((f - 1) * C + (c - 1)) * KH + (ki - 1)) * KW + (kj - 1)) + 1
								acc += xs[xi + 1] * ws[wi]
							end
						end
					end
					ys[(((f - 1) * Ho + (i - 1)) * Wo + (j - 1)) * B + b] = acc
				end
			end
		end
	end
	task.synchronize()
	return Tape.makeNode(Y, { X, W }, function(gy: Types.Tensor)
		local Ho, Wo = (H - KH) + 1, (Wd - KW) + 1
		-- dX
		if X._requiresGrad then
			X._grad = X._grad or Tensor.zeros({ C, H, Wd, B }, X._dtype, false)
			task.desynchronize()
			for b = 1, B do
				for f = 1, F do
					for i = 1, Ho do
						for j = 1, Wo do
							local gyv = gy._storage[(((f - 1) * Ho + (i - 1)) * Wo + (j - 1)) * B + b]
							if gyv ~= 0 then
								for c = 1, C do
									for ki = 1, KH do
										for kj = 1, KW do
											local xi = (((c - 1) * H + (i + ki - 2)) * Wd + (j + kj - 2)) * B + b
											local wi = ((((f - 1) * C + (c - 1)) * KH + (ki - 1)) * KW + (kj - 1)) + 1
											X._grad._storage[xi + 1] += gyv * W._storage[wi]
										end
									end
								end
							end
						end
					end
				end
			end
			task.synchronize()
		end
		-- dW
		if W._requiresGrad then
			W._grad = W._grad or Tensor.zeros({ F, C, KH, KW }, W._dtype, false)
			task.desynchronize()
			for b = 1, B do
				for f = 1, F do
					for i = 1, Ho do
						for j = 1, Wo do
							local gyv = gy._storage[(((f - 1) * Ho + (i - 1)) * Wo + (j - 1)) * B + b]
							if gyv ~= 0 then
								for c = 1, C do
									for ki = 1, KH do
										for kj = 1, KW do
											local xi = (((c - 1) * H + (i + ki - 2)) * Wd + (j + kj - 2)) * B + b
											local wi = ((((f - 1) * C + (c - 1)) * KH + (ki - 1)) * KW + (kj - 1)) + 1
											W._grad._storage[wi] += gyv * X._storage[xi + 1]
										end
									end
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
