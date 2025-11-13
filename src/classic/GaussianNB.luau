--!optimize 2
return function(C: number)
	local mu = {}
	local var = {}
	local prior = table.create(C, 0)
	local fitted = false
	local self
	self = {
		fit = function(_, X, y: { number })
			local D, N = X._shape[1], X._shape[2]
			task.desynchronize()
			for c = 1, C do
				mu[c] = table.create(D, 0)
				var[c] = table.create(D, 0)
				prior[c] = 0
			end
			for n = 1, N do
				prior[y[n]] += 1
			end
			for c = 1, C do
				if prior[c] == 0 then
					prior[c] = 1
				end
			end
			for c = 1, C do
				for d = 1, D do
					local s = 0
					for n = 1, N do
						if y[n] == c then
							s += X._storage[(d - 1) * N + n]
						end
					end
					mu[c][d] = s / prior[c]
				end
			end
			for c = 1, C do
				for d = 1, D do
					local s = 0
					for n = 1, N do
						if y[n] == c then
							local dx = X._storage[(d - 1) * N + n] - mu[c][d]
							s += dx * dx
						end
					end
					var[c][d] = math.max(s / prior[c], 1e-6)
				end
			end
			for c = 1, C do
				prior[c] = prior[c] / N
			end
			task.synchronize()
			fitted = true
		end,
		predict = function(_, X)
			assert(fitted, "GaussianNB: fit first")
			local D, N = X._shape[1], X._shape[2]
			local out = table.create(N, 1)
			task.desynchronize()
			for n = 1, N do
				local best = -1 / 0
				local arg = 1
				for c = 1, C do
					local logp = math.log(prior[c])
					for d = 1, D do
						local m = mu[c][d]
						local v = var[c][d]
						local x = X._storage[(d - 1) * N + n]
						logp += -0.5 * math.log(2 * math.pi * v) - (x - m) * (x - m) / (2 * v)
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
