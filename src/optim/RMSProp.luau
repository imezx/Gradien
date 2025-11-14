--!optimize 2
export type Optimizer = { step: (self: Optimizer) -> (), zeroGrad: (self: Optimizer) -> () }

return function(params: { any }, lr: number, decay: number?, eps: number?)
	decay = decay or 0.99
	eps = eps or 1e-8
	local avg2: { [any]: { [number]: number } } = {}
	local self: Optimizer = {
		step = function(_)
			task.desynchronize()
			for _, p in params do
				if p._grad then
					local ps, gs = p._storage, p._grad._storage
					local a2 = avg2[p]
					if not a2 then
						a2 = table.create(#ps, 0)
						avg2[p] = a2
					end
					for i = 1, #ps do
						a2[i] = (decay :: number) * a2[i] + (1 - (decay :: number)) * gs[i] * gs[i]
						ps[i] -= lr * gs[i] / (math.sqrt(a2[i]) + (eps :: number))
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
