--!optimize 2
local Tensor = require("../Tensor")
local Util = require("../Util")
local assert = Util.Assert

return function()
	local mu: { number } = nil
	local sigma: { number } = nil
	local self
	self = {
		fit = function(_, X)
			task.desynchronize()
			local D, N = X._shape[1], X._shape[2]
			mu = table.create(D, 0)
			sigma = table.create(D, 1)
			for d = 1, D do
				local s, s2 = 0, 0
				for n = 1, N do
					local v = X._storage[(d - 1) * N + n]
					s += v
					s2 += v * v
				end
				local m = s / N
				local v = math.max(s2 / N - m * m, 1e-9)
				mu[d] = m
				sigma[d] = math.sqrt(v)
			end
			task.synchronize()
		end,
		transform = function(_, X)
			assert(mu and sigma, "StandardScaler: fit first")
			task.desynchronize()
			local D, N = X._shape[1], X._shape[2]
			local Y = Tensor.zeros(X._shape)
			for d = 1, D do
				local m = (mu :: { number })[d]
				local s = (sigma :: { number })[d]
				for n = 1, N do
					Y._storage[(d - 1) * N + n] = (X._storage[(d - 1) * N + n] - m) / s
				end
			end
			task.synchronize()
			return Y
		end,
	}
	return self
end
