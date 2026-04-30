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
end

-- Element-wise binary ops

function Math.add(A: Types.Tensor, B: Types.Tensor): Types.Tensor
	checkShapes(A, B, "add")
	task.desynchronize()
	local C = Tensor.empty(A._shape, A._dtype, A._requiresGrad or B._requiresGrad)
	local as, bs, cs = A._storage, B._storage, C._storage
	for i = 1, #as do cs[i] = as[i] + bs[i] end
	task.synchronize()
	return C
end

function Math.sub(A: Types.Tensor, B: Types.Tensor): Types.Tensor
	checkShapes(A, B, "sub")
	task.desynchronize()
	local C = Tensor.empty(A._shape, A._dtype, A._requiresGrad or B._requiresGrad)
	local as, bs, cs = A._storage, B._storage, C._storage
	for i = 1, #as do cs[i] = as[i] - bs[i] end
	task.synchronize()
	return C
end

function Math.mul(A: Types.Tensor, B: Types.Tensor): Types.Tensor
	checkShapes(A, B, "mul")
	task.desynchronize()
	local C = Tensor.empty(A._shape, A._dtype, A._requiresGrad or B._requiresGrad)
	local as, bs, cs = A._storage, B._storage, C._storage
	for i = 1, #as do cs[i] = as[i] * bs[i] end
	task.synchronize()
	return C
end

function Math.div(A: Types.Tensor, B: Types.Tensor): Types.Tensor
	checkShapes(A, B, "div")
	task.desynchronize()
	local C = Tensor.empty(A._shape, A._dtype, A._requiresGrad or B._requiresGrad)
	local as, bs, cs = A._storage, B._storage, C._storage
	for i = 1, #as do cs[i] = as[i] / bs[i] end
	task.synchronize()
	return C
end

-- Scalar ops

function Math.scalarAdd(A: Types.Tensor, s: number): Types.Tensor
	task.desynchronize()
	local C = Tensor.empty(A._shape, A._dtype, A._requiresGrad)
	local as, cs = A._storage, C._storage
	for i = 1, #as do cs[i] = as[i] + s end
	task.synchronize()
	return C
end

function Math.scalarSub(A: Types.Tensor, s: number): Types.Tensor
	task.desynchronize()
	local C = Tensor.empty(A._shape, A._dtype, A._requiresGrad)
	local as, cs = A._storage, C._storage
	for i = 1, #as do cs[i] = as[i] - s end
	task.synchronize()
	return C
end

function Math.scalarMul(A: Types.Tensor, s: number): Types.Tensor
	task.desynchronize()
	local C = Tensor.empty(A._shape, A._dtype, A._requiresGrad)
	local as, cs = A._storage, C._storage
	for i = 1, #as do cs[i] = as[i] * s end
	task.synchronize()
	return C
end

function Math.scalarDiv(A: Types.Tensor, s: number): Types.Tensor
	task.desynchronize()
	local C = Tensor.empty(A._shape, A._dtype, A._requiresGrad)
	local as, cs = A._storage, C._storage
	for i = 1, #as do cs[i] = as[i] / s end
	task.synchronize()
	return C
end

-- Unary ops

function Math.neg(A: Types.Tensor): Types.Tensor
	task.desynchronize()
	local C = Tensor.empty(A._shape, A._dtype, A._requiresGrad)
	local as, cs = A._storage, C._storage
	for i = 1, #as do cs[i] = -as[i] end
	task.synchronize()
	return C
end

function Math.abs(A: Types.Tensor): Types.Tensor
	task.desynchronize()
	local C = Tensor.empty(A._shape, A._dtype, A._requiresGrad)
	local as, cs = A._storage, C._storage
	for i = 1, #as do cs[i] = math.abs(as[i]) end
	task.synchronize()
	return C
end

function Math.exp(A: Types.Tensor): Types.Tensor
	task.desynchronize()
	local C = Tensor.empty(A._shape, A._dtype, A._requiresGrad)
	local as, cs = A._storage, C._storage
	for i = 1, #as do cs[i] = math.exp(as[i]) end
	task.synchronize()
	return C
end

function Math.log(A: Types.Tensor): Types.Tensor
	task.desynchronize()
	local C = Tensor.empty(A._shape, A._dtype, A._requiresGrad)
	local as, cs = A._storage, C._storage
	for i = 1, #as do cs[i] = math.log(as[i]) end
	task.synchronize()
	return C
end

function Math.sqrt(A: Types.Tensor): Types.Tensor
	task.desynchronize()
	local C = Tensor.empty(A._shape, A._dtype, A._requiresGrad)
	local as, cs = A._storage, C._storage
	for i = 1, #as do cs[i] = math.sqrt(as[i]) end
	task.synchronize()
	return C
end

function Math.pow(A: Types.Tensor, exp: number): Types.Tensor
	task.desynchronize()
	local C = Tensor.empty(A._shape, A._dtype, A._requiresGrad)
	local as, cs = A._storage, C._storage
	for i = 1, #as do cs[i] = as[i] ^ exp end
	task.synchronize()
	return C
end

function Math.clamp(A: Types.Tensor, min: number, max: number): Types.Tensor
	local C = Tensor.empty(A._shape, A._dtype, A._requiresGrad)
	local as, cs = A._storage, C._storage
	for i = 1, #as do cs[i] = math.clamp(as[i], min, max) end
	return C
end

-- Reductions

function Math.sum(A: Types.Tensor): number
	task.desynchronize()
	local as = A._storage
	local acc = 0
	for i = 1, #as do acc += as[i] end
	task.synchronize()
	return acc
end

function Math.mean(A: Types.Tensor): number
	return Math.sum(A) / #A._storage
end

function Math.max(A: Types.Tensor): number
	local as = A._storage
	local acc = as[1]
	for i = 2, #as do
		if as[i] > acc then acc = as[i] end
	end
	return acc
end

function Math.min(A: Types.Tensor): number
	local as = A._storage
	local acc = as[1]
	for i = 2, #as do
		if as[i] < acc then acc = as[i] end
	end
	return acc
end

return Math
