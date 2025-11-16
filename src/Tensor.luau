--!optimize 2
local Tensor = {}
Tensor.__index = Tensor

local Util = require("./Util")
local Types = require("./Types")
local Tape -- lazy-load

export type Tensor = typeof(setmetatable({} :: Types.Tensor, Tensor))

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

local function makeRangeView(storage: { number }, startIndex: number, length: number)
	local view = {
		_base = storage,
		_offset = startIndex - 1,
		_len = length,
	}
	return setmetatable(view, {
		__len = function(tbl)
			return tbl._len
		end,
		__index = function(tbl, key)
			if type(key) ~= "number" then
				return rawget(tbl, key)
			end
			if key < 1 or key > tbl._len then
				error("Tensor view index out of range")
			end
			return tbl._base[tbl._offset + key]
		end,
		__newindex = function(tbl, key, value)
			if type(key) ~= "number" then
				rawset(tbl, key, value)
				return
			end
			if key < 1 or key > tbl._len then
				error("Tensor view index out of range")
			end
			tbl._base[tbl._offset + key] = value
		end,
	})
end

local function makeTranspose2DView(storage: { number }, rows: number, cols: number)
	local total = rows * cols
	local view = {
		_base = storage,
		_rows = rows,
		_cols = cols,
		_len = total,
	}
	return setmetatable(view, {
		__len = function(tbl)
			return tbl._len
		end,
		__index = function(tbl, key)
			if type(key) ~= "number" then
				return rawget(tbl, key)
			end
			if key < 1 or key > tbl._len then
				error("Tensor transpose view index out of range")
			end
			local colsOut = tbl._rows
			local zero = key - 1
			local rowOut = math.floor(zero / colsOut)
			local colOut = zero % colsOut
			local baseRow = colOut + 1
			local baseCol = rowOut + 1
			local baseIdx = (baseRow - 1) * tbl._cols + baseCol
			return tbl._base[baseIdx]
		end,
		__newindex = function(tbl, key, value)
			if type(key) ~= "number" then
				rawset(tbl, key, value)
				return
			end
			if key < 1 or key > tbl._len then
				error("Tensor transpose view index out of range")
			end
			local colsOut = tbl._rows
			local zero = key - 1
			local rowOut = math.floor(zero / colsOut)
			local colOut = zero % colsOut
			local baseRow = colOut + 1
			local baseCol = rowOut + 1
			local baseIdx = (baseRow - 1) * tbl._cols + baseCol
			tbl._base[baseIdx] = value
		end,
	})
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
	return setmetatable({
		_storage = storage,
		_shape = shapeClone,
		_dtype = dtype or "f64",
		_requiresGrad = requiresGrad == true,
		_size = sizeOverride or #storage,
		_strides = strides,
		_parents = nil,
		_backprop = nil,
		_grad = nil,
	}, Tensor)
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

--@param shape {number}
--@param dtype string?
--@param requiresGrad boolean?
--@return Tensor
function Tensor.zeros(shape: { number }, dtype: Types.DType?, requiresGrad: boolean?): Tensor
	dtype = dtype or "f64"
	local n = Util.sizeFromShape(shape)
	return newTensor(table.create(n, 0), shape, dtype, requiresGrad, n)
end

--@param shape {number}
--@param dtype string?
--@param requiresGrad boolean?
--@return Tensor
function Tensor.ones(shape: { number }, dtype: Types.DType?, requiresGrad: boolean?): Tensor
	dtype = dtype or "f64"
	local n = Util.sizeFromShape(shape)
	return newTensor(table.create(n, 1), shape, dtype, requiresGrad, n)
end

--@param shape {number}
--@param dtype string?
--@param requiresGrad boolean?
function Tensor.empty(shape: { number }, dtype: Types.DType?, requiresGrad: boolean?): Tensor
	dtype = dtype or "f64"
	local n = Util.sizeFromShape(shape)
	return newTensor(table.create(n), shape, dtype, requiresGrad, n)
end

--@param data {number}
--@param shape {number}
--@param dtype string?
--@param requiresGrad boolean?
--@return Tensor
function Tensor.fromArray(
	data: { number },
	shape: { number },
	dtype: Types.DType?,
	requiresGrad: boolean?
): Tensor
	local n = Util.sizeFromShape(shape)
	assert(#data == n, ("Tensor.fromArray: got %d elements, expected %d"):format(#data, n))
	dtype = dtype or "f64"
	return newTensor(table.clone(data), shape, dtype, requiresGrad, n)
end

--@param self Tensor
--@param newShape {number}
--@return Tensor
function Tensor.reshape(self: Tensor, newShape: { number }): Tensor
	local size = ensureSize(self)
	assert(Util.sizeFromShape(newShape) == size, "reshape: element count mismatch")
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

--@param self Tensor
--@param dim number -- dimension to slice (1-based)
--@param startIdx number -- inclusive start index
--@param endIdx number? -- inclusive end index (default: full dimension)
--@param step number? -- step > 0 (default: 1)
--@return Tensor
function Tensor.slice(self: Tensor, dim: number, startIdx: number, endIdx: number?, step: number?): Tensor
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

	local outSize = Util.sizeFromShape(outShape)
	local inStrides = ensureStrides(self)
	local outStrides = computeStrides(outShape)
	local useView = (dim == 1 and st == 1)
	local outStorage
	local indexMap
	local contiguousOffset
	if useView then
		local chunk = outSize / length
		local offset = (s - 1) * chunk
		contiguousOffset = offset
		outStorage = makeRangeView(self._storage, offset + 1, outSize)
	else
		outStorage = table.create(outSize)
		local map = self._requiresGrad and table.create(outSize) or nil
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
			if map then
				map[flatOut] = storageIndex
			end
		end
		task.synchronize()
		indexMap = map
	end
	local out = newTensor(outStorage, outShape, self._dtype, self._requiresGrad, outSize)
	if not self._requiresGrad then
		return out
	end
	if not Tape then
		Tape = require("./autograd/Tape") :: any
	end
	local shapeCopy = table.clone(shape)
	local mapCopy = indexMap
	local contiguousOffsetCopy = contiguousOffset
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
				gsrc[mapCopy[i]] += gys[i]
			end
		else
			local start = contiguousOffsetCopy :: number
			for i = 1, #gys do
				gsrc[start + i] += gys[i]
			end
		end
		task.synchronize()
	end)
end

--@param self Tensor
--@param dim1 number? -- first dimension (1-based)
--@param dim2 number? -- second dimension (1-based)
--@return Tensor
function Tensor.transpose(self: Tensor, dim1: number?, dim2: number?): Tensor
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

	local useView = (rank == 2 and ((d1 == 1 and d2 == 2) or (d1 == 2 and d2 == 1)))
	local dst
	local indexMap
	if useView then
		dst = makeTranspose2DView(self._storage, shape[1], shape[2])
	else
		dst = table.create(outSize)
		local map = self._requiresGrad and table.create(outSize) or nil
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
			if map then
				map[outIndex] = flatIn
			end
		end
		task.synchronize()
		indexMap = map
	end
	local out = newTensor(dst, outShape, self._dtype, self._requiresGrad, outSize)
	if not self._requiresGrad then
		return out
	end
	if not Tape then
		Tape = require("./autograd/Tape") :: any
	end
	local shapeCopy = table.clone(shape)
	local mapCopy = indexMap
	local viewRows = useView and shape[1] or nil
	local viewCols = useView and shape[2] or nil
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
				gsrc[mapCopy[i]] += gys[i]
			end
		else
			local rows = viewRows :: number
			local cols = viewCols :: number
			for row = 1, rows do
				local inBase = (row - 1) * cols
				for col = 1, cols do
					local outIdx = (col - 1) * rows + row
					gsrc[inBase + col] += gys[outIdx]
				end
			end
		end
		task.synchronize()
	end)
end

--@param self Tensor
--@param dim number -- dimension to narrow (1-based)
--@param startIdx number -- inclusive start index
--@param length number -- number of elements to keep along that dim
--@return Tensor
function Tensor.narrow(self: Tensor, dim: number, startIdx: number, length: number): Tensor
	assert(length >= 1, "narrow: length must be >= 1")
	local endIdx = startIdx + length - 1
	return Tensor.slice(self, dim, startIdx, endIdx, 1)
end

function Tensor.detach(self: Tensor): Tensor
	return newTensor(self._storage, self._shape, self._dtype, false, ensureSize(self))
end

--@param self Tensor
function Tensor.noGrad(self: Tensor): ()
	self._requiresGrad = false
	self._grad = nil
end

return Tensor
