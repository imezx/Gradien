--!optimize 2
local Types = require("../Types")

return function(params: { Types.Tensor }, lr: number, momentum: number?, nesterov: boolean?)
	momentum = momentum or 0
	local vel: { [Types.Tensor]: { [number]: number } } = {}
	local step = 0
	local self = {
		step = function(_)
			step += 1
			task.desynchronize()
			for _, p in params do
				if p._grad then
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
				end
			end
			task.synchronize()
		end,
		zeroGrad = function(_)
			for _, p in params do
				p._grad = nil
			end
		end,
		setLr = function(_, newLr: number)
			lr = newLr
		end,
		getState = function()
			return {
				lr = lr,
				momentum = momentum,
				step = step,
				velocity = vel,
			}
		end,
		setState = function(
			_,
			state: {
				lr: number,
				momentum: number,
				step: number,
				velocity: { [Types.Tensor]: { [number]: number } },
			}
		)
			if state.lr ~= nil then
				lr = state.lr
			end
			if state.momentum ~= nil then
				momentum = state.momentum
			end
			if state.step ~= nil then
				step = state.step
			end
			if state.velocity ~= nil then
				vel = state.velocity
			end
		end,
	}
	return self
end
