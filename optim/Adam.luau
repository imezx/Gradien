--!optimize 2
export type Optimizer = { step: (self: Optimizer) -> (), zeroGrad: (self: Optimizer) -> () }

return function(params: { any }, lr: number, b1: number?, b2: number?, eps: number?)
	b1 = b1 or 0.9
	b2 = b2 or 0.999
	eps = eps or 1e-8
	local m: { [any]: { [number]: number } } = {}
	local v: { [any]: { [number]: number } } = {}
	local t = 0
	local self: Optimizer = {
		step = function(_)
			t += 1
			local b1t = (b1 :: number) ^ t
			local b2t = (b2 :: number) ^ t
			for _, p in params do
				if p._grad then
					task.desynchronize()
					local ps, gs = p._storage, p._grad._storage
					local mi = m[p]
					if not mi then
						mi = table.create(#ps, 0)
						m[p] = mi
					end
					local vi = v[p]
					if not vi then
						vi = table.create(#ps, 0)
						v[p] = vi
					end
					for i = 1, #ps do
						local g = gs[i]
						mi[i] = (b1 :: number) * mi[i] + (1 - (b1 :: number)) * g
						vi[i] = (b2 :: number) * vi[i] + (1 - (b2 :: number)) * g * g
						local mHat = mi[i] / (1 - b1t)
						local vHat = vi[i] / (1 - b2t)
						ps[i] -= lr * (mHat / (math.sqrt(vHat) + (eps :: number)))
					end
					task.synchronize()
				end
			end
		end,
		zeroGrad = function(_)
			for _, p in params do
				p._grad = nil
			end
		end,
	}
	return self
end
