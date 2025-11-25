--!native
--!optimize 2
local Tensor = require("../Tensor")
local Util = require("../Util")
local assert = Util.Assert

return function(K: number, iters: number)
	iters = iters or 20
	local centers: any = nil
	local function initPlusPlus(X)
		local D, N = X._shape[1], X._shape[2]
		local C = Tensor.zeros({ D, K })
		local first = math.random(1, N)
		for i = 1, D do
			C._storage[(i - 1) * K + 1] = X._storage[(i - 1) * N + first]
		end
		local dist = table.create(N, math.huge)
		task.desynchronize()
		for c = 2, K do
			for n = 1, N do
				local best = dist[n]
				for cc = 1, c - 1 do
					local s = 0
					for d = 1, D do
						local dx = X._storage[(d - 1) * N + n] - C._storage[(d - 1) * K + cc]
						s += dx * dx
					end
					if s < best then
						best = s
					end
				end
				dist[n] = best
			end
			local sum = 0
			for n = 1, N do
				sum += dist[n]
			end
			local r = math.random() * math.max(sum, 1e-9)
			local acc = 0
			local pick = N
			for n = 1, N do
				acc += dist[n]
				if acc >= r then
					pick = n
					break
				end
			end
			for d = 1, D do
				C._storage[(d - 1) * K + c] = X._storage[(d - 1) * N + pick]
			end
		end
		task.synchronize()
		return C
	end
	local self
	self = {
		fit = function(_, X)
			local D, N = X._shape[1], X._shape[2]
			centers = initPlusPlus(X)
			local labels = table.create(N, 1)
			task.desynchronize()
			for _ = 1, iters do
				for n = 1, N do
					local best = math.huge
					local li = 1
					for k = 1, K do
						local s = 0
						for d = 1, D do
							local dx = X._storage[(d - 1) * N + n] - centers._storage[(d - 1) * K + k]
							s += dx * dx
						end
						if s < best then
							best = s
							li = k
						end
					end
					labels[n] = li
				end
				local counts = table.create(K, 0)
				local newC = Tensor.zeros({ D, K })
				for n = 1, N do
					local k = labels[n]
					counts[k] += 1
					for d = 1, D do
						newC._storage[(d - 1) * K + k] += X._storage[(d - 1) * N + n]
					end
				end
				for k = 1, K do
					local c = math.max(counts[k], 1)
					for d = 1, D do
						newC._storage[(d - 1) * K + k] /= c
					end
				end
				centers = newC
			end
			task.synchronize()
			return centers
		end,
		predict = function(_, X)
			assert(centers, "KMeans: fit first")
			local D, N = X._shape[1], X._shape[2]
			local labels = table.create(N, 1)
			task.desynchronize()
			for n = 1, N do
				local best = math.huge
				local li = 1
				for k = 1, K do
					local s = 0
					for d = 1, D do
						local dx = X._storage[(d - 1) * N + n] - centers._storage[(d - 1) * K + k]
						s += dx * dx
					end
					if s < best then
						best = s
						li = k
					end
				end
				labels[n] = li
			end
			task.synchronize()
			return labels
		end,
		centers = function(_)
			return centers
		end,
	}
	return self
end
