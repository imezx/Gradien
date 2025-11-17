--!optimize 2
local Tensor = require("../Tensor")
local Util = require("../Util")
local assert = Util.Assert

return function(a: number, b: number)
	a = a or 0
	b = b or 1
	local lo: { number }? = nil
	local hi: { number }? = nil
	local self
	self = {
		fit = function(_, X)
			task.desynchronize()
			local D, N = X._shape[1], X._shape[2]
			lo = table.create(D, 1 / 0)
			hi = table.create(D, -1 / 0)
			for d = 1, D do
				for n = 1, N do
					local v = X._storage[(d - 1) * N + n]
					if v < (lo :: { number })[d] then
						(lo :: { number })[d] = v
					end
					if v > (hi :: { number })[d] then
						(hi :: { number })[d] = v
					end
				end
			end
			task.synchronize()
		end,
		transform = function(_, X)
			assert(lo and hi, "MinMaxScaler: fit first")
			task.desynchronize()
			local D, N = X._shape[1], X._shape[2]
			local Y = Tensor.zeros(X._shape)
			for d = 1, D do
				local l = (lo :: { number })[d]
				local h = (hi :: { number })[d]
				local range = (h - l ~= 0) and (h - l) or 1
				for n = 1, N do
					Y._storage[(d - 1) * N + n] = a + (X._storage[(d - 1) * N + n] - l) * (b - a) / range
				end
			end
			task.synchronize()
			return Y
		end,
	}
	return self
end
