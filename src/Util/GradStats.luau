--!native
--!optimize 2
local Types = require("../Types")
local GradStats = {}

local function stats(t: Types.Tensor): (number, number, number, number)
	local n = #t._storage
	if n == 0 then
		return 0, 0, 0, 0
	end
	task.desynchronize()
	local minv, maxv = math.huge, -math.huge
	local sum, sumsq = 0, 0
	for i = 1, n do
		local v = t._storage[i]
		if v < minv then
			minv = v
		end
		if v > maxv then
			maxv = v
		end
		sum += v
		sumsq += v * v
	end
	local mean = sum / n
	local var = math.max(sumsq / n - mean * mean, 0)
	local std = math.sqrt(var)
	task.synchronize()
	return minv, maxv, mean, std
end

function GradStats.forModel(model: Types.Module): { { min: number, max: number, mean: number, std: number, n: number } }
	local ps = model:parameters()
	local out = {}
	for i, p in ps do
		if p._grad then
			local minv, maxv, mean, std = stats(p._grad)
			out[i] = {
				min = minv,
				max = maxv,
				mean = mean,
				std = std,
				n = #p._grad._storage,
			}
		end
	end
	return out
end

return GradStats
