--!native
--!optimize 2
local Tape = require("../autograd/Tape")
local Types = require("../Types")
local FN = {}

function FN.relu(x: Types.Tensor): Types.Tensor
	return Tape.relu(x)
end

function FN.linear(w: Types.Tensor, b: Types.Tensor, x: Types.Tensor): Types.Tensor
	task.desynchronize()
	local out = Tape.matmul(w, x)
	local B = x._shape[2]
	for i = 1, w._shape[1] do
		local bi = b._storage[i]
		for j = 1, B do
			out._storage[(i - 1) * B + j] += bi
		end
	end
	task.synchronize()
	return out
end

return FN
