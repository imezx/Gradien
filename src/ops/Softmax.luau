--!optimize 2
local Tensor = require("../Tensor")
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
	return stableSoftmax(logits)
end

return SoftmaxOps

