--!optimize 2
local Tensor = require("../Tensor")
local Types = require("../Types")
local Math = {}

local function checkShapes(A: Types.Tensor, B: Types.Tensor, op: string)
	if #A._storage ~= #B._storage then
		error(
			`Math.{op}: Shape mismatch. A has {#A._storage} elements, B has {#B._storage}. Broadcasting is not currently supported.`
		)
	end
	return
end

--@param A Tensor
--@param B Tensor
--@return Tensor
function Math.add(A: Types.Tensor, B: Types.Tensor): Types.Tensor
	checkShapes(A, B, "add")
	task.desynchronize()
	local C = Tensor.empty(A._shape, A._dtype, A._requiresGrad or B._requiresGrad)
	local as, bs, cs = A._storage, B._storage, C._storage
	for i = 1, #as do
		cs[i] = as[i] + bs[i]
	end
	task.synchronize()
	return C
end

--@param A Tensor
--@param B Tensor
--@return Tensor
function Math.sub(A: Types.Tensor, B: Types.Tensor): Types.Tensor
	checkShapes(A, B, "sub")
	task.desynchronize()
	local C = Tensor.empty(A._shape, A._dtype, A._requiresGrad or B._requiresGrad)
	local as, bs, cs = A._storage, B._storage, C._storage
	for i = 1, #as do
		cs[i] = as[i] - bs[i]
	end
	task.synchronize()
	return C
end

--@param A Tensor
--@param B Tensor
--@return Tensor
function Math.mul(A: Types.Tensor, B: Types.Tensor): Types.Tensor
	checkShapes(A, B, "mul")
	task.desynchronize()
	local C = Tensor.empty(A._shape, A._dtype, A._requiresGrad or B._requiresGrad)
	local as, bs, cs = A._storage, B._storage, C._storage
	for i = 1, #as do
		cs[i] = as[i] * bs[i]
	end
	task.synchronize()
	return C
end

--@param A Tensor
--@param s number
--@return Tensor
function Math.scalarAdd(A: Types.Tensor, s: number): Types.Tensor
	task.desynchronize()
	local C = Tensor.empty(A._shape, A._dtype, A._requiresGrad)
	local as, cs = A._storage, C._storage
	for i = 1, #as do
		cs[i] = as[i] + s
	end
	task.synchronize()
	return C
end

--@param A Tensor
--@param s number
--@return Tensor
function Math.scalarMul(A: Types.Tensor, s: number): Types.Tensor
	task.desynchronize()
	local C = Tensor.empty(A._shape, A._dtype, A._requiresGrad)
	local as, cs = A._storage, C._storage
	for i = 1, #as do
		cs[i] = as[i] * s
	end
	task.synchronize()
	return C
end

return Math
