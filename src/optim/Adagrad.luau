--!optimize 2
export type Optimizer = { step: (self: Optimizer) -> (), zeroGrad: (self: Optimizer) -> () }

return function(params: { any }, lr: number, eps: number?)
	eps = eps or 1e-8
	local G: { [any]: { [number]: number } } = {}
	local self: Optimizer = {
		step = function(_)
			task.desynchronize()
			for _, p in params do
				if p._grad then
					local ps, gs = p._storage, p._grad._storage
					local Gi = G[p]
					if not Gi then
						Gi = table.create(#ps, 0)
						G[p] = Gi
					end
					for i = 1, #ps do
						Gi[i] += gs[i] * gs[i]
						ps[i] -= lr * gs[i] / (math.sqrt(Gi[i]) + (eps :: number))
					end
				end
			end
			task.synchronize()
		end,
		zeroGrad = function(_)
			for _, p in params do
				p._grad = nil
			end
		end,
	}
	return self
end
