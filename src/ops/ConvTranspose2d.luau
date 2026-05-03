--!native
--!optimize 2
local Tensor = require("../Tensor")
local Tape = require("../autograd/Tape")
local BLAS = require("../ops/BLAS")
local Types = require("../Types")
local Util = require("../Util")
local assert = Util.Assert

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
		_requiresGrad = tensor._requiresGrad,
		_parents = nil,
		_backprop = nil,
		_grad = nil,
		_size = nil,
	}
end

local function reshapeInPlace(tensor: Types.Tensor, shape: { number }): Types.Tensor
	tensor._shape = copyShape(shape)
	return tensor
end

local function transposeMatrix(matrix: Types.Tensor): Types.Tensor
	local rows, cols = matrix._shape[1], matrix._shape[2]
	local out = Tensor.zeros({ cols, rows }, matrix._requiresGrad)
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

local function col2im(
	cols: Types.Tensor,
	C: number,
	H: number,
	Wd: number,
	B: number,
	KH: number,
	KW: number,
	Ho: number,
	Wo: number
): Types.Tensor
	local patch = C * KH * KW
	local out = Tensor.zeros({ C, H, Wd, B }, false)
	local xs = out._storage
	local cs = cols._storage
	local HWd_B = H * Wd * B
	local Wd_B = Wd * B
	task.desynchronize()
	local col = 0
	for i = 1, Ho do
		for j = 1, Wo do
			for b = 1, B do
				local base = col * patch
				col += 1
				local offset = 0
				local b_minus_1 = b - 1
				for c = 1, C do
					local c_offset = (c - 1) * HWd_B
					for ki = 0, KH - 1 do
						local row_part = (i + ki - 1) * Wd_B
						for kj = 0, KW - 1 do
							local col_part = (j + kj - 1) * B
							local xi = c_offset + row_part + col_part + b_minus_1

							xs[xi + 1] += cs[base + offset + 1]
							offset += 1
						end
					end
				end
			end
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
	local cols = Tensor.zeros({ patch, Ho * Wo * B }, false)
	local xs = X._storage
	local cs = cols._storage
	local HWd_B = H * Wd * B
	local Wd_B = Wd * B
	task.desynchronize()
	local col = 0
	for i = 1, Ho do
		for j = 1, Wo do
			for b = 1, B do
				local base = col * patch
				col += 1
				local offset = 0
				local b_minus_1 = b - 1
				for c = 1, C do
					local c_offset = (c - 1) * HWd_B
					for ki = 0, KH - 1 do
						local row_part = (i + ki - 1) * Wd_B
						for kj = 0, KW - 1 do
							local col_part = (j + kj - 1) * B
							local xi = c_offset + row_part + col_part + b_minus_1
							cs[base + offset + 1] = xs[xi + 1]
							offset += 1
						end
					end
				end
			end
		end
	end
	task.synchronize()
	return cols
end

local function addInto(dst: Types.Tensor, src: Types.Tensor)
	task.desynchronize()
	local ds, ss = dst._storage, src._storage
	for i = 1, #ds do
		ds[i] += ss[i]
	end
	task.synchronize()
end

return function(X: Types.Tensor, W: Types.Tensor): Types.Tensor
	local C_in, H, Wd, B = X._shape[1], X._shape[2], X._shape[3], X._shape[4]
	local Cin_w, C_out, KH, KW = W._shape[1], W._shape[2], W._shape[3], W._shape[4]
	assert(C_in == Cin_w, "convTranspose2d: in_channels mismatch")
	task.desynchronize()
	local Ho = H + KH - 1
	local Wo = Wd + KW - 1
	local X_flat = makeMatrixView(X, C_in, H * Wd * B)
	local W_flat = makeMatrixView(W, C_in, C_out * KH * KW)
	local W_T = transposeMatrix(W_flat) -- {C_out * KH * KW, C_in}
	local cols = BLAS.matmul(W_T, X_flat) -- {C_out * KH * KW, H * Wd * B}
	local Y = col2im(cols, C_out, Ho, Wo, B, KH, KW, H, Wd)
	task.synchronize()
	return Tape.makeNode(Y, { X, W }, function(gy: Types.Tensor)
		local gy_cols = im2col(gy, C_out, Ho, Wo, B, KH, KW, H, Wd)
		if X._requiresGrad then
			if not X._grad then
				X._grad = Tensor.zeros({ C_in, H, Wd, B }, false)
			end
			local dX_flat = BLAS.matmul(W_flat, gy_cols)
			local dX = reshapeInPlace(dX_flat, { C_in, H, Wd, B })
			addInto(X._grad :: Types.Tensor, dX)
		end
		if W._requiresGrad then
			if not W._grad then
				W._grad = Tensor.zeros({ C_in, C_out, KH, KW }, false)
			end
			local gy_cols_T = transposeMatrix(gy_cols)
			local dW_flat = BLAS.matmul(X_flat, gy_cols_T)
			local dW = reshapeInPlace(dW_flat, { C_in, C_out, KH, KW })
			addInto(W._grad :: Types.Tensor, dW)
		end
	end)
end
