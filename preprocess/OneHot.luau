--!optimize 2
local Tensor = require("../Tensor")
return function(C: number)
	return function(idx: { number })
		task.desynchronize()
		local B = #idx
		local Y = Tensor.zeros({ C, B })
		for j = 1, B do
			Y._storage[(idx[j] - 1) * B + j] = 1
		end
		task.synchronize()
		return Y
	end
end
