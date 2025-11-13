--!optimize 2
local Tensor = require("../Tensor")
local Types = require("../Types")
local Math = {}

--@param A Tensor
--@param B Tensor
--@return Tensor
function Math.add(A: Types.Tensor, B: Types.Tensor): Types.Tensor
	task.desynchronize()
	local C = Tensor.zeros(A._shape, A._dtype, A._requiresGrad or B._requiresGrad)
	for i = 1, #A._storage do
		C._storage[i] = A._storage[i] + B._storage[i]
	end
	task.synchronize()
	return C
end

--@param A Tensor
--@param B Tensor
--@return Tensor
function Math.sub(A: Types.Tensor, B: Types.Tensor): Types.Tensor
	task.desynchronize()
	local C = Tensor.zeros(A._shape, A._dtype, A._requiresGrad or B._requiresGrad)
	for i = 1, #A._storage do
		C._storage[i] = A._storage[i] - B._storage[i]
	end
	task.synchronize()
	return C
end

--@param A Tensor
--@param B Tensor
--@return Tensor
function Math.mul(A: Types.Tensor, B: Types.Tensor): Types.Tensor
	task.desynchronize()
	local C = Tensor.zeros(A._shape, A._dtype, A._requiresGrad or B._requiresGrad)
	for i = 1, #A._storage do
		C._storage[i] = A._storage[i] * B._storage[i]
	end
	task.synchronize()
	return C
end

--@param A Tensor
--@param s number
--@return Tensor
function Math.scalarAdd(A: Types.Tensor, s: number): Types.Tensor
	task.desynchronize()
	local C = Tensor.zeros(A._shape, A._dtype, A._requiresGrad)
	for i = 1, #A._storage do
		C._storage[i] = A._storage[i] + s
	end
	task.synchronize()
	return C
end

--@param A Tensor
--@param s number
--@return Tensor
function Math.scalarMul(A: Types.Tensor, s: number): Types.Tensor
	task.desynchronize()
	local C = Tensor.zeros(A._shape, A._dtype, A._requiresGrad)
	for i = 1, #A._storage do
		C._storage[i] = A._storage[i] * s
	end
	task.synchronize()
	return C
end

return Math
