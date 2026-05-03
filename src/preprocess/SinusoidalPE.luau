--!native
--!optimize 2
local Tensor = require("../Tensor")

--@param x Tensor -- {D, B*T}
--@param T number -- sequence length
--@return Tensor -- new tensor {D, B*T} with PE added
return function(x, T: number)
	local D, BT = x._shape[1], x._shape[2]
	local B = math.floor(BT / T)
	local out = Tensor.zeros(x._shape, x._requiresGrad)
	task.desynchronize()
	local pe = {}
	for pos = 1, T do
		pe[pos] = table.create(D, 0.0)
		for i = 1, D, 2 do
			local div = math.exp(((i - 1) / D) * math.log(10000))
			pe[pos][i] = math.sin((pos - 1) / div)
			if i + 1 <= D then
				pe[pos][i + 1] = math.cos((pos - 1) / div)
			end
		end
	end
	for b = 1, B do
		local base = (b - 1) * T
		for t = 1, T do
			for i = 1, D do
				local col = base + t
				out._storage[(i - 1) * BT + col] = x._storage[(i - 1) * BT + col] + pe[t][i]
			end
		end
	end
	task.synchronize()
	return out
end
