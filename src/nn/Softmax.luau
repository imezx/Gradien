--!optimize 2
local Tensor = require("../Tensor")
local Types = require("../Types")

local S = {}

function S.forward(logits: Types.Tensor): Types.Tensor
	task.desynchronize()
	local C, B = logits._shape[1], logits._shape[2]
	local P = Tensor.zeros({ C, B }, logits._dtype, logits._requiresGrad)
	for j = 1, B do
		local m = -1 / 0
		for i = 1, C do
			local v = logits._storage[(i - 1) * B + j]
			if v > m then
				m = v
			end
		end
		local sum = 0
		for i = 1, C do
			local e = math.exp(logits._storage[(i - 1) * B + j] - m)
			P._storage[(i - 1) * B + j] = e
			sum += e
		end
		for i = 1, C do
			P._storage[(i - 1) * B + j] /= sum
		end
	end
	task.synchronize()
	return P
end

return S
