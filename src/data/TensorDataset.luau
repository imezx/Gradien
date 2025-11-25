--!native
--!optimize 2
local Tensor = require("../Tensor")
local Types = require("../Types")
return function(X: Types.Tensor, Y: Types.Tensor): Types.Dataset
	local D, N = X._shape[1], X._shape[2]
	local function colSlice(startIdx: number, endIdx: number): (Types.Tensor, Types.Tensor)
		local B = endIdx - startIdx + 1
		local XB = Tensor.zeros({ D, B }, X._dtype, X._requiresGrad)
		for d = 1, D do
			local base = (d - 1) * N
			for b = 1, B do
				XB._storage[(d - 1) * B + b] = X._storage[base + (startIdx + b - 1)]
			end
		end
		local yB = table.create(B, 1)
		for b = 1, B do
			yB[b] = Y[startIdx + b - 1]
		end
		return XB, yB
	end
	local function colGather(indices: { number }): (Types.Tensor, Types.Tensor)
		local B = #indices
		local XB = Tensor.zeros({ D, B }, X._dtype, X._requiresGrad)
		for d = 1, D do
			local base = (d - 1) * N
			for b = 1, B do
				XB._storage[(d - 1) * B + b] = X._storage[base + indices[b]]
			end
		end
		local yB = table.create(B, 1)
		for b = 1, B do
			yB[b] = Y[indices[b]]
		end
		return XB, yB
	end
	return {
		size = function(self): number
			return N
		end,
		at = function(self, i: number): (Types.Tensor, Types.Tensor)
			return colSlice(i, i)
		end,
		slice = function(self, i: number, j: number): (Types.Tensor, Types.Tensor)
			return colSlice(i, j)
		end,
		batch = function(self, indices: { number }): (Types.Tensor, Types.Tensor)
			return colGather(indices)
		end,
	}
end
