--!optimize 2
local Tensor = require("../Tensor")
local Types = require("../Types")

-- Matrix multiply: A{M,K} x B{K,N} -> C{M,N}
--@param A Tensor
--@param B Tensor
--@return Tensor
local function matmul(A: Types.Tensor, B: Types.Tensor): Types.Tensor
	local M, K = A._shape[1], A._shape[2]
	local K2, N = B._shape[1], B._shape[2]
	assert(K == K2, "matmul: inner dims mismatch")
	task.desynchronize()
	local C = Tensor.zeros({ M, N }, A._dtype, A._requiresGrad or B._requiresGrad)
	local as, bs, cs = A._storage, B._storage, C._storage
	for i = 1, M do
		for j = 1, N do
			local acc = 0
			for k = 1, K do
				acc += as[(i - 1) * K + k] * bs[(k - 1) * N + j]
			end
			cs[(i - 1) * N + j] = acc
		end
	end
	task.synchronize()
	return C
end

return { matmul = matmul }
