--!optimize 2
local Tensor = {}

local Util = require("./Util")
local Types = require("./Types")
local Tape -- lazy-load

local TensorMethods = {}
local function attachMethods(t)
	for name, fn in TensorMethods do
		t[name] = fn
	end
end

export type Tensor = Types.Tensor

local sizeFromShape = Util.sizeFromShape
local assert = Util.Assert
local function computeStrides(shape: { number }): { number }
	local rank = #shape
	local strides = table.create(rank)
	local acc: number = 1
	task.desynchronize()
	for i = rank, 1, -1 do
		strides[i] = acc
		acc *= shape[i]
	end
	task.synchronize()
	return strides
end

local function newTensor(
	storage: { number },
	shape: { number },
	dtype: Types.DType?,
	requiresGrad: boolean?,
	sizeOverride: number?
): Tensor
	local shapeClone = table.clone(shape)
	local strides = computeStrides(shapeClone)
	local t: Types.Tensor = {
		_storage = storage,
		_shape = shapeClone,
		_dtype = dtype or "f64",
		_requiresGrad = requiresGrad == true,
		_size = sizeOverride or #storage,
		_strides = strides,
		_parents = nil,
		_backprop = nil,
		_grad = nil,
	}
	attachMethods(t :: any)
	return t :: any
end

local function ensureSize(t: Types.Tensor): number
	local size = t._size
	if not size then
		size = #t._storage
		t._size = size
	end
	return size
end

local function ensureStrides(t: Types.Tensor): { number }
	local strides = t._strides
	if not strides then
		strides = computeStrides(t._shape)
		t._strides = strides
	end
	return strides
end

function Tensor.zeros(shape: { number }, dtype: Types.DType?, requiresGrad: boolean?): Tensor
	dtype = dtype or "f64"
	local n = sizeFromShape(shape)
	return newTensor(table.create(n, 0), shape, dtype, requiresGrad, n)
end

function Tensor.ones(shape: { number }, dtype: Types.DType?, requiresGrad: boolean?): Tensor
	dtype = dtype or "f64"
	local n = sizeFromShape(shape)
	return newTensor(table.create(n, 1), shape, dtype, requiresGrad, n)
end

function Tensor.empty(shape: { number }, dtype: Types.DType?, requiresGrad: boolean?): Tensor
	dtype = dtype or "f64"
	local n = sizeFromShape(shape)
	return newTensor(table.create(n), shape, dtype, requiresGrad, n)
end

function Tensor.fromArray(data: { number }, shape: { number }, dtype: Types.DType?, requiresGrad: boolean?): Tensor
	local n = sizeFromShape(shape)
	assert(#data == n, ("Tensor.fromArray: got %d elements, expected %d"):format(#data, n))
	dtype = dtype or "f64"
	return newTensor(table.clone(data), shape, dtype, requiresGrad, n)
end

local function reshape(self: Tensor, newShape: { number }): Tensor
	local size = ensureSize(self)
	assert(sizeFromShape(newShape) == size, "reshape: element count mismatch")
	local out = newTensor(self._storage, newShape, self._dtype, self._requiresGrad, size)

	if not self._requiresGrad then
		return out
	end
	if not Tape then
		Tape = require("./autograd/Tape") :: any
	end
	return Tape.makeNode(out, { self }, function(gy: Types.Tensor)
		if self._requiresGrad then
			self._grad = self._grad or Tensor.zeros(self._shape, self._dtype, false)
			task.desynchronize()
			local gs, ys = self._grad._storage, gy._storage
			for i = 1, #ys do
				gs[i] += ys[i]
			end
			task.synchronize()
		end
	end)
end

local function slice(self: Tensor, dim: number, startIdx: number, endIdx: number?, step: number?): Tensor
	local shape = self._shape
	local rank = #shape
	assert(dim >= 1 and dim <= rank, "slice: dim out of range")

	local sizeDim = shape[dim]
	local s = startIdx
	local e = endIdx or sizeDim
	local st = step or 1

	assert(st > 0, "slice: step must be > 0")
	assert(s >= 1 and s <= sizeDim, "slice: startIdx out of bounds")
	assert(e >= 1 and e <= sizeDim, "slice: endIdx out of bounds")
	assert(e >= s, "slice: endIdx must be >= startIdx")

	local length = math.floor((e - s) / st) + 1
	local outShape = table.clone(shape)
	outShape[dim] = length

	local outSize = sizeFromShape(outShape)
	local inStrides = ensureStrides(self)
	local outStrides = computeStrides(outShape)

	local outStorage = table.create(outSize)
	local indexMap = self._requiresGrad and table.create(outSize) or nil

	task.desynchronize()
	local coord = table.create(rank, 1)
	local srcStorage = self._storage
	for flatOut = 1, outSize do
		local idx0 = flatOut - 1
		for d = 1, rank do
			local stride = outStrides[d]
			local v = math.floor(idx0 / stride)
			coord[d] = v + 1
			idx0 = idx0 % stride
		end
		local posDim = s + (coord[dim] - 1) * st
		local flatIn0 = 0
		for d = 1, rank do
			local cd = coord[d]
			if d == dim then
				cd = posDim
			end
			flatIn0 += (cd - 1) * inStrides[d]
		end
		local storageIndex = flatIn0 + 1
		outStorage[flatOut] = srcStorage[storageIndex]
		if indexMap then
			indexMap[flatOut] = storageIndex
		end
	end
	task.synchronize()

	local out = newTensor(outStorage, outShape, self._dtype, self._requiresGrad, outSize)
	if not self._requiresGrad then
		return out
	end
	if not Tape then
		Tape = require("./autograd/Tape") :: any
	end
	local shapeCopy = table.clone(shape)
	local mapCopy = indexMap
	local dtype = self._dtype
	return Tape.makeNode(out, { self }, function(gy: Types.Tensor)
		if not self._requiresGrad then
			return
		end
		self._grad = self._grad or Tensor.zeros(shapeCopy, dtype, false)
		task.desynchronize()
		local gsrc = self._grad._storage
		local gys = gy._storage
		if mapCopy then
			for i = 1, #gys do
				local dest = mapCopy[i]
				if dest then
					gsrc[dest] += gys[i]
				end
			end
		end
		task.synchronize()
	end)
end

local function transpose(self: Tensor, dim1: number?, dim2: number?): Tensor
	local shape = self._shape
	local rank = #shape

	local d1 = dim1
	local d2 = dim2
	if d1 == nil and d2 == nil then
		assert(rank == 2, "transpose: default dims only valid for rank-2 tensors")
		d1, d2 = 1, 2
	else
		assert(d1 ~= nil and d2 ~= nil, "transpose: both dim1 and dim2 must be provided")
	end

	assert(d1 >= 1 and d1 <= rank, "transpose: dim1 out of range")
	assert(d2 >= 1 and d2 <= rank, "transpose: dim2 out of range")
	assert(d1 ~= d2, "transpose: dim1 and dim2 must be different")

	local perm = table.create(rank)
	for i = 1, rank do
		perm[i] = i
	end
	perm[d1], perm[d2] = perm[d2], perm[d1]

	local outShape = table.create(rank)
	for i = 1, rank do
		outShape[i] = shape[perm[i]]
	end

	local size = ensureSize(self)
	local outSize = size
	local inStrides = ensureStrides(self)
	local outStrides = computeStrides(outShape)

	local dst = table.create(outSize)
	local indexMap = self._requiresGrad and table.create(outSize) or nil
	task.desynchronize()
	local coordIn = table.create(rank, 1)
	for flatIn = 1, size do
		local idx0 = flatIn - 1
		for d = 1, rank do
			local stride = inStrides[d]
			local v = math.floor(idx0 / stride)
			coordIn[d] = v + 1
			idx0 = idx0 % stride
		end
		local flatOut0 = 0
		for outDim = 1, rank do
			local inDim = perm[outDim]
			local c = coordIn[inDim]
			flatOut0 += (c - 1) * outStrides[outDim]
		end
		local outIndex = flatOut0 + 1
		dst[outIndex] = self._storage[flatIn]
		if indexMap then
			indexMap[outIndex] = flatIn
		end
	end
	task.synchronize()

	local out = newTensor(dst, outShape, self._dtype, self._requiresGrad, outSize)
	if not self._requiresGrad then
		return out
	end
	if not Tape then
		Tape = require("./autograd/Tape") :: any
	end
	local shapeCopy = table.clone(shape)
	local mapCopy = indexMap
	local dtype = self._dtype

	return Tape.makeNode(out, { self }, function(gy: Types.Tensor)
		if not self._requiresGrad then
			return
		end
		self._grad = self._grad or Tensor.zeros(shapeCopy, dtype, false)
		task.desynchronize()
		local gsrc = self._grad._storage
		local gys = gy._storage
		if mapCopy then
			for i = 1, #gys do
				local srcIdx = mapCopy[i]
				if srcIdx then
					gsrc[srcIdx] += gys[i]
				end
			end
		end
		task.synchronize()
	end)
end

local function narrow(self: Tensor, dim: number, startIdx: number, length: number): Tensor
	assert(length >= 1, "narrow: length must be >= 1")
	local endIdx = startIdx + length - 1
	return slice(self, dim, startIdx, endIdx, 1)
end

local function detach(self: Tensor): Tensor
	return newTensor(self._storage, self._shape, self._dtype, false, ensureSize(self))
end

local function noGrad(self: Tensor): ()
	self._requiresGrad = false
	self._grad = nil
end

TensorMethods.reshape = reshape
Tensor.reshape = reshape

TensorMethods.slice = slice
Tensor.slice = slice

TensorMethods.transpose = transpose
Tensor.transpose = transpose

TensorMethods.narrow = narrow
Tensor.narrow = narrow

TensorMethods.detach = detach
Tensor.detach = detach

TensorMethods.noGrad = noGrad
Tensor.noGrad = noGrad

return Tensor
