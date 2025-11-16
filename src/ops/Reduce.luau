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
	local out = Tensor.empty({ 1, B }, X._dtype, X._requiresGrad)
	local xs, ys = X._storage, out._storage
	for j = 1, B do
		local acc = 0
		local idx = j
		for _ = 1, D do
			acc += xs[idx]
			idx += B
		end
		ys[j] = acc
	end
	task.synchronize()
	return out
end

-- Mean per column (shape {D,B} -> {1,B})
--@param X Tensor
--@return Tensor
function Reduce.meanCols(X: Types.Tensor): Types.Tensor
	task.desynchronize()
	local D, B = X._shape[1], X._shape[2]
	local out = Tensor.empty({ 1, B }, X._dtype, X._requiresGrad)
	local xs, ys = X._storage, out._storage
	local invD = 1 / D
	for j = 1, B do
		local acc = 0
		local idx = j
		for _ = 1, D do
			acc += xs[idx]
			idx += B
		end
		ys[j] = acc * invD
	end
	task.synchronize()
	return out
end

return Reduce
