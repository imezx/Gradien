--!native
--!optimize 2
local Tensor = require("../Tensor")
local Tape = require("../autograd/Tape")
local Types = require("../Types")

local SoftmaxOps = {}

local function stableSoftmax(logits: Types.Tensor, out: Types.Tensor)
	local src = logits._storage
	local dst = out._storage
	local C, B = logits._shape[1], logits._shape[2]
	task.desynchronize()
	for j = 1, B do
		local colStart = (1 - 1) * B + j
		local maxVal = -math.huge
		local idx = colStart -- pointer
		for i = 1, C do
			local v = src[idx]
			if v > maxVal then
				maxVal = v
			end
			idx += B
		end
		local sum = 0
		idx = colStart
		for i = 1, C do
			local e = math.exp(src[idx] - maxVal)
			dst[idx] = e
			sum += e
			idx += B
		end
		-- normalize
		local invSum = 1 / sum
		idx = colStart
		for i = 1, C do
			dst[idx] *= invSum
			idx += B
		end
	end
	task.synchronize()
end

function SoftmaxOps.forward(logits: Types.Tensor): Types.Tensor
	local P = Tensor.zeros(logits._shape, logits._dtype, logits._requiresGrad)
	stableSoftmax(logits, P)
	if not logits._requiresGrad then
		return P
	end
	return Tape.makeNode(P, { logits }, function(gy: Types.Tensor)
		if not logits._requiresGrad then
			return
		end
		if not logits._grad then
			logits._grad = Tensor.zeros(logits._shape, logits._dtype, false)
		end
		task.desynchronize()
		local grads = logits._grad._storage
		local probs = P._storage
		local gys = gy._storage
		local C, B = logits._shape[1], logits._shape[2]
		-- dx_i = y_i * (dy_i - dot(y, dy))
		for j = 1, B do
			local colStart = (1 - 1) * B + j
			local dot = 0
			local idx = colStart
			for i = 1, C do
				dot += probs[idx] * gys[idx]
				idx += B
			end
			idx = colStart
			for i = 1, C do
				grads[idx] += probs[idx] * (gys[idx] - dot)
				idx += B
			end
		end
		task.synchronize()
	end)
end

return SoftmaxOps
