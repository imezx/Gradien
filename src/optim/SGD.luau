--!optimize 2
export type Optimizer = { step: (self: Optimizer) -> (), zeroGrad: (self: Optimizer) -> () }

return function(params: { any }, lr: number, momentum: number?, nesterov: boolean?)
	momentum = momentum or 0
	local vel: { [any]: { [number]: number } } = {}
	local self: Optimizer = {
		step = function(_)
			for _, p in params do
				if p._grad then
					task.desynchronize()
					local ps, gs = p._storage, p._grad._storage
					local v = vel[p]
					if not v then
						v = table.create(#ps, 0)
						vel[p] = v
					end
					for i = 1, #ps do
						v[i] = (momentum :: number) * v[i] + gs[i]
						local upd = nesterov and ((momentum :: number) * v[i] + gs[i]) or v[i]
						ps[i] -= lr * upd
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
