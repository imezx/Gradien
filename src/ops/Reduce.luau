--!optimize 2
local Tensor = require("../Tensor")
local Types = require("../Types")
local Reduce = {}

-- Sum per column (shape {D,B} -> {1,B})
--@param X Tensor
--@return Tensor
function Reduce.sumCols(X: Types.Tensor): Types.Tensor
	task.desynchronize()
	local D, B = X._shape[1], X._shape[2]
	local out = Tensor.zeros({ 1, B }, X._dtype, X._requiresGrad)
	for j = 1, B do
		local acc = 0
		for i = 1, D do
			acc += X._storage[(i - 1) * B + j]
		end
		out._storage[j] = acc
	end
	task.synchronize()
	return out
end

-- Mean per column (shape {D,B} -> {1,B})
--@param X Tensor
--@return Tensor
function Reduce.meanCols(X: Types.Tensor): Types.Tensor
	task.desynchronize()
	local D = X._shape[1]
	local s = Reduce.sumCols(X)
	for j = 1, #s._storage do
		s._storage[j] /= D
	end
	task.synchronize()
	return s
end

return Reduce
