--!optimize 2
local Tensor = require("../Tensor")
local Tape = require("../autograd/Tape")
local Types = require("../Types")
local BLAS = require("../ops/BLAS")

local function copyShape(shape: { number }): { number }
	local out = table.create(#shape)
	for i = 1, #shape do
		out[i] = shape[i]
	end
	return out
end

local function makeMatrixView(tensor: Types.Tensor, rows: number, cols: number): Types.Tensor
	return {
		_storage = tensor._storage,
		_shape = { rows, cols },
		_dtype = tensor._dtype,
		_requiresGrad = tensor._requiresGrad,
		_parents = nil,
		_backprop = nil,
		_grad = nil,
	}
end

local function reshapeInPlace(tensor: Types.Tensor, shape: { number }): Types.Tensor
	tensor._shape = copyShape(shape)
	return tensor
end

local function transposeMatrix(matrix: Types.Tensor): Types.Tensor
	local rows, cols = matrix._shape[1], matrix._shape[2]
	local out = Tensor.zeros({ cols, rows }, matrix._dtype, matrix._requiresGrad)
	task.desynchronize()
	local os, ms = out._storage, matrix._storage
	for r = 1, rows do
		local rowOffset = (r - 1) * cols
		for c = 1, cols do
			os[(c - 1) * rows + r] = ms[rowOffset + c]
		end
	end
	task.synchronize()
	return out
end

local function im2col(
	X: Types.Tensor,
	C: number,
	H: number,
	Wd: number,
	B: number,
	KH: number,
	KW: number,
	Ho: number,
	Wo: number
)
	local patch = C * KH * KW
	local cols = Tensor.zeros({ patch, Ho * Wo * B }, X._dtype, false)
	local xs = X._storage
	local cs = cols._storage
	task.desynchronize()
	local col = 0
	for i = 1, Ho do
		for j = 1, Wo do
			for b = 1, B do
				col += 1
				local base = (col - 1) * patch
				local offset = 0
				for c = 1, C do
					for ki = 1, KH do
						for kj = 1, KW do
							offset += 1
							local xi = (((c - 1) * H + (i + ki - 2)) * Wd + (j + kj - 2)) * B + b
							cs[base + offset] = xs[xi + 1]
						end
					end
				end
			end
		end
	end
	task.synchronize()
	return cols
end

local function col2im(
	cols: Types.Tensor,
	C: number,
	H: number,
	Wd: number,
	B: number,
	KH: number,
	KW: number,
	Ho: number,
	Wo: number,
	dtype: Types.DType
): Types.Tensor
	local patch = C * KH * KW
	local out = Tensor.zeros({ C, H, Wd, B }, dtype, false)
	local xs = out._storage
	local cs = cols._storage
	task.desynchronize()
	local col = 0
	for i = 1, Ho do
		for j = 1, Wo do
			for b = 1, B do
				col += 1
				local base = (col - 1) * patch
				local offset = 0
				for c = 1, C do
					for ki = 1, KH do
						for kj = 1, KW do
							offset += 1
							local xi = (((c - 1) * H + (i + ki - 2)) * Wd + (j + kj - 2)) * B + b
							xs[xi + 1] += cs[base + offset]
						end
					end
				end
			end
		end
	end
	task.synchronize()
	return out
end

local function addInto(dst: Types.Tensor, src: Types.Tensor)
	task.desynchronize()
	local ds, ss = dst._storage, src._storage
	for i = 1, #ds do
		ds[i] += ss[i]
	end
	task.synchronize()
end

---@param X Tensor
---@param W Tensor -- {F,C,KH,KW}
---@return Tensor -- {F,Ho,Wo,B}
return function(X: Types.Tensor, W: Types.Tensor): Types.Tensor
	local C, H, Wd, B = X._shape[1], X._shape[2], X._shape[3], X._shape[4]
	local F, Cw, KH, KW = W._shape[1], W._shape[2], W._shape[3], W._shape[4]
	assert(C == Cw, "conv2d: channel mismatch")
	local Ho = H - KH + 1
	local Wo = Wd - KW + 1
	assert(Ho > 0 and Wo > 0, "conv2d: kernel larger than input")

	local patch = C * KH * KW
	local columns = Ho * Wo * B

	local cols = im2col(X, C, H, Wd, B, KH, KW, Ho, Wo)
	local weightView = makeMatrixView(W, F, patch)
	local yMat = BLAS.matmul(weightView, cols)
	local Y = reshapeInPlace(yMat, { F, Ho, Wo, B })

	return Tape.makeNode(Y, { X, W }, function(gy: Types.Tensor)
		local gyMat = makeMatrixView(gy, F, columns)

		if W._requiresGrad then
			if not W._grad then
				W._grad = Tensor.zeros({ F, C, KH, KW }, W._dtype, false)
			end
			local colsT = transposeMatrix(cols)
			local dW = BLAS.matmul(gyMat, colsT)
			local gradW = W._grad :: Types.Tensor
			local gradView = makeMatrixView(gradW, F, patch)
			addInto(gradView, dW)
		end

		if X._requiresGrad then
			if not X._grad then
				X._grad = Tensor.zeros({ C, H, Wd, B }, X._dtype, false)
			end
			local weightT = transposeMatrix(weightView)
			local dXcols = BLAS.matmul(weightT, gyMat)
			local dX = col2im(dXcols, C, H, Wd, B, KH, KW, Ho, Wo, X._dtype)
			local gradX = X._grad :: Types.Tensor
			addInto(gradX, dX)
		end
	end)
end
