--!optimize 2
local Tensor = require("../Tensor")
local Tape = require("../autograd/Tape")
local Types = require("../Types")

local SoftmaxOps = {}

local function stableSoftmax(logits: Types.Tensor): Types.Tensor
	task.desynchronize()
	local D, B = logits._shape[1], logits._shape[2]
	local P = Tensor.zeros({ D, B }, logits._dtype, logits._requiresGrad)
	for j = 1, B do
		local m = -1 / 0
		for i = 1, D do
			local v = logits._storage[(i - 1) * B + j]
			if v > m then
				m = v
			end
		end
		local sum = 0
		for i = 1, D do
			local e = math.exp(logits._storage[(i - 1) * B + j] - m)
			P._storage[(i - 1) * B + j] = e
			sum += e
		end
		for i = 1, D do
			P._storage[(i - 1) * B + j] /= sum
		end
	end
	task.synchronize()
	return P
end

function SoftmaxOps.forward(logits: Types.Tensor): Types.Tensor
	local P = stableSoftmax(logits)
	if not logits._requiresGrad then
		return P
	end
	return Tape.makeNode(P, { logits }, function(gy: Types.Tensor)
		if not logits._requiresGrad then
			return
		end
		logits._grad = logits._grad or Tensor.zeros(logits._shape, logits._dtype, false)
		task.desynchronize()
		local gradStore = logits._grad._storage
		local probStore = P._storage
		local gyStore = gy._storage
		local D, B = logits._shape[1], logits._shape[2]
		for j = 1, B do
			local dot = 0
			for i = 1, D do
				local idx = (i - 1) * B + j
				dot += probStore[idx] * gyStore[idx]
			end
			for i = 1, D do
				local idx = (i - 1) * B + j
				gradStore[idx] += probStore[idx] * (gyStore[idx] - dot)
			end
		end
		task.synchronize()
	end)
end

return SoftmaxOps

