--!optimize 2
local Tensor = {}
Tensor.__index = Tensor

local Util = require("./Util")
local Types = require("./Types")
local Tape -- lazy-load

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

--@param shape {number}
--@param dtype string?
--@param requiresGrad boolean?
--@return Tensor
function Tensor.zeros(shape: { number }, dtype: Types.DType?, requiresGrad: boolean?): Types.Tensor
	dtype = dtype or "f64"
	local n = Util.sizeFromShape(shape)
	local self = setmetatable({
		_storage = table.create(n, 0),
		_shape = table.clone(shape),
		_dtype = dtype,
		_requiresGrad = requiresGrad == true,
	}, Tensor)
	return self
end

--@param shape {number}
--@param dtype string?
--@param requiresGrad boolean?
--@return Tensor
function Tensor.ones(shape: { number }, dtype: Types.DType?, requiresGrad: boolean?): Types.Tensor
	dtype = dtype or "f64"
	local n = Util.sizeFromShape(shape)
	local self = setmetatable({
		_storage = table.create(n, 1),
		_shape = table.clone(shape),
		_dtype = dtype,
		_requiresGrad = requiresGrad == true,
	}, Tensor)
	return self
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
): Types.Tensor
	local n = Util.sizeFromShape(shape)
	assert(#data == n, ("Tensor.fromArray: got %d elements, expected %d"):format(#data, n))
	dtype = dtype or "f64"
	local self = setmetatable({
		_storage = table.clone(data),
		_shape = table.clone(shape),
		_dtype = dtype,
		_requiresGrad = requiresGrad == true,
	}, Tensor)
	return self
end

--@param self Tensor
--@param newShape {number}
--@return Tensor
function Tensor.reshape(self: Types.Tensor, newShape: { number }): Types.Tensor
	assert(Util.sizeFromShape(newShape) == #self._storage, "reshape: element count mismatch")
	local out = setmetatable({
		_storage = self._storage,
		_shape = table.clone(newShape),
		_dtype = self._dtype,
		_requiresGrad = self._requiresGrad,
		_parents = nil,
		_backprop = nil,
		_grad = nil,
	}, Tensor)

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

-- @param self Tensor
-- @param dim number -- dimension to slice (1-based)
-- @param startIdx number -- inclusive start index
-- @param endIdx number? -- inclusive end index (default: full dimension)
-- @param step number? -- step > 0 (default: 1)
-- @return Tensor
function Tensor.slice(self: Types.Tensor, dim: number, startIdx: number, endIdx: number?, step: number?): Types.Tensor
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
	local inStrides = computeStrides(shape)
	local outStrides = computeStrides(outShape)
	local useView = (dim == 1 and st == 1)
	local outStorage
	if useView then
		local chunk = outSize / length
		local offset = (s - 1) * chunk + 1
		outStorage = makeRangeView(self._storage, offset, outSize)
	else
		outStorage = table.create(outSize)
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
			outStorage[flatOut] = srcStorage[flatIn0 + 1]
		end
		task.synchronize()
	end
	local out = setmetatable({
		_storage = outStorage,
		_shape = outShape,
		_dtype = self._dtype,
		_requiresGrad = self._requiresGrad,
		_parents = nil,
		_backprop = nil,
		_grad = nil,
	}, Tensor)
	if not self._requiresGrad then
		return out
	end
	if not Tape then
		Tape = require("./autograd/Tape") :: any
	end
	local shapeCopy = table.clone(shape)
	local inStridesCopy = computeStrides(shapeCopy)
	local outStridesCopy = computeStrides(outShape)
	local dimCopy = dim
	local sCopy = s
	local stCopy = st
	return Tape.makeNode(out, { self }, function(gy: Types.Tensor)
		if not self._requiresGrad then
			return
		end
		self._grad = self._grad or Tensor.zeros(shapeCopy, self._dtype, false)
		task.desynchronize()
		local gsrc = self._grad._storage
		local gys = gy._storage
		local coord = table.create(rank, 1)
		local total = #gys
		for flatOut = 1, total do
			local idx0 = flatOut - 1
			for d = 1, rank do
				local stride = outStridesCopy[d]
				local v = math.floor(idx0 / stride)
				coord[d] = v + 1
				idx0 = idx0 % stride
			end
			local posDim = sCopy + (coord[dimCopy] - 1) * stCopy
			local flatIn0 = 0
			for d = 1, rank do
				local cd = coord[d]
				if d == dimCopy then
					cd = posDim
				end
				flatIn0 += (cd - 1) * inStridesCopy[d]
			end
			gsrc[flatIn0 + 1] += gys[flatOut]
		end
		task.synchronize()
	end)
end

-- @param self Tensor
-- @param dim1 number? -- first dimension (1-based)
-- @param dim2 number? -- second dimension (1-based)
-- @return Tensor
function Tensor.transpose(self: Types.Tensor, dim1: number?, dim2: number?): Types.Tensor
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

	local inSize = Util.sizeFromShape(shape)
	local outSize = Util.sizeFromShape(outShape)
	assert(inSize == outSize, "transpose: internal size mismatch")

	local inStrides = computeStrides(shape)
	local outStrides = computeStrides(outShape)

	local useView = (rank == 2 and ((d1 == 1 and d2 == 2) or (d1 == 2 and d2 == 1)))
	local dst
	if useView then
		dst = makeTranspose2DView(self._storage, shape[1], shape[2])
	else
		dst = table.create(outSize)
		task.desynchronize()
		local coordIn = table.create(rank, 1)
		for flatIn = 1, inSize do
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
			dst[flatOut0 + 1] = self._storage[flatIn]
		end
		task.synchronize()
	end
	local out = setmetatable({
		_storage = dst,
		_shape = outShape,
		_dtype = self._dtype,
		_requiresGrad = self._requiresGrad,
		_parents = nil,
		_backprop = nil,
		_grad = nil,
	}, Tensor)
	if not self._requiresGrad then
		return out
	end
	if not Tape then
		Tape = require("./autograd/Tape") :: any
	end
	local shapeCopy = table.clone(shape)
	local permCopy = table.clone(perm)
	local inStridesCopy = computeStrides(shapeCopy)
	local outStridesCopy = computeStrides(outShape)

	return Tape.makeNode(out, { self }, function(gy: Types.Tensor)
		if not self._requiresGrad then
			return
		end
		self._grad = self._grad or Tensor.zeros(shapeCopy, self._dtype, false)
		task.desynchronize()
		local gsrc = self._grad._storage
		local gys = gy._storage
		local coordIn = table.create(rank, 1)
		local total = #gsrc

		for flatIn = 1, total do
			local idx0 = flatIn - 1
			for d = 1, rank do
				local stride = inStridesCopy[d]
				local v = math.floor(idx0 / stride)
				coordIn[d] = v + 1
				idx0 = idx0 % stride
			end
			local flatOut0 = 0
			for outDim = 1, rank do
				local inDim = permCopy[outDim]
				local c = coordIn[inDim]
				flatOut0 += (c - 1) * outStridesCopy[outDim]
			end
			gsrc[flatIn] += gys[flatOut0 + 1]
		end
		task.synchronize()
	end)
end

-- @param self Tensor
-- @param dim number -- dimension to narrow (1-based)
-- @param startIdx number -- inclusive start index
-- @param length number -- number of elements to keep along that dim
-- @return Tensor
function Tensor.narrow(self: Types.Tensor, dim: number, startIdx: number, length: number): Types.Tensor
	assert(length >= 1, "narrow: length must be >= 1")
	local endIdx = startIdx + length - 1
	return Tensor.slice(self, dim, startIdx, endIdx, 1)
end

function Tensor.detach(self: Types.Tensor): Types.Tensor
	local out = setmetatable({
		_storage = self._storage,
		_shape = table.clone(self._shape),
		_dtype = self._dtype,
		_requiresGrad = false,
		_parents = nil,
		_backprop = nil,
		_grad = nil,
	}, Tensor)
	return out
end

return Tensor
