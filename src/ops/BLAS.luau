--!optimize 2
local Tensor = require("../Tensor")
local Types = require("../Types")

--@param A Tensor
--@param B Tensor
--@return Tensor
local function matmul(A: Types.Tensor, B: Types.Tensor): Types.Tensor
	local M, K = A._shape[1], A._shape[2]
	local K2, N = B._shape[1], B._shape[2]
	assert(K == K2, "matmul: inner dims mismatch")
	local C = Tensor.zeros({ M, N }, A._dtype, A._requiresGrad or B._requiresGrad)
	task.desynchronize()
	local as, bs, cs = A._storage, B._storage, C._storage
	for i = 1, M do
		local aRowOffset = (i - 1) * K
		local cRowOffset = (i - 1) * N
		for k = 1, K do
			local aVal = as[aRowOffset + k]
			if aVal ~= 0 then
				local bRowOffset = (k - 1) * N
				local baseC = cRowOffset
				local limit = N - (N % 4)
				local j = 1
				while j <= limit do
					local idxC = baseC + j
					local idxB = bRowOffset + j
					cs[idxC] += aVal * bs[idxB]
					cs[idxC + 1] += aVal * bs[idxB + 1]
					cs[idxC + 2] += aVal * bs[idxB + 2]
					cs[idxC + 3] += aVal * bs[idxB + 3]
					j += 4
				end
				for jj = limit + 1, N do
					cs[baseC + jj] += aVal * bs[bRowOffset + jj]
				end
			end
		end
	end
	task.synchronize()
	return C
end

return {
	matmul = matmul,
}
