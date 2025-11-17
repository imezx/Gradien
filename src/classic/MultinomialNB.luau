--!optimize 2
local Util = require("../Util")
local assert = Util.Assert

return function(C: number, alpha: number)
	alpha = alpha or 1.0
	local classCount = table.create(C, 0)
	local featureCount = {}
	local fitted = false
	local self
	self = {
		fit = function(_, X, y)
			local D, N = X._shape[1], X._shape[2]
			task.desynchronize()
			for c = 1, C do
				featureCount[c] = table.create(D, 0)
				classCount[c] = 0
			end
			for n = 1, N do
				local c = y[n]
				classCount[c] += 1
				for d = 1, D do
					featureCount[c][d] += math.max(0, X._storage[(d - 1) * N + n])
				end
			end
			task.synchronize()
			fitted = true
		end,
		predict = function(_, X)
			assert(fitted, "MultinomialNB: fit first")
			local D, N = X._shape[1], X._shape[2]
			local out = table.create(N, 1)
			task.desynchronize()
			for n = 1, N do
				local best = -1 / 0
				local arg = 1
				for c = 1, C do
					local sum = 0
					for d = 1, D do
						sum += featureCount[c][d]
					end
					local logp = math.log((classCount[c] + alpha) / (N + C * alpha))
					for d = 1, D do
						local p = (featureCount[c][d] + alpha) / (sum + D * alpha)
						local x = math.max(0, X._storage[(d - 1) * N + n])
						if p > 0 then
							logp += x * math.log(p)
						end
					end
					if logp > best then
						best = logp
						arg = c
					end
				end
				out[n] = arg
			end
			task.synchronize()
			return out
		end,
	}
	return self
end
