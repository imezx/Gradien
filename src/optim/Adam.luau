--!native
--!optimize 2
local Types = require("../Types")

return function(params: { Types.Tensor }, lr: number, b1: number?, b2: number?, eps: number?)
	b1 = b1 or 0.9
	b2 = b2 or 0.999
	eps = eps or 1e-8
	local m: { [Types.Tensor]: { [number]: number } } = {}
	local v: { [Types.Tensor]: { [number]: number } } = {}
	local t = 0
	local self = {
		step = function(_)
			t += 1
			local b1t = (b1 :: number) ^ t
			local b2t = (b2 :: number) ^ t
			task.desynchronize()
			for _, p in params do
				if p._grad then
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
		getState = function(): {
			lr: number,
			b1: number,
			b2: number,
			eps: number,
			t: number,
			m: { [Types.Tensor]: { [number]: number } },
			v: { [Types.Tensor]: { [number]: number } },
		}
			return { lr = lr, b1 = b1, b2 = b2, eps = eps, t = t, m = m, v = v }
		end,
		setState = function(
			_,
			state: {
				lr: number,
				b1: number,
				b2: number,
				eps: number,
				t: number,
				m: { [Types.Tensor]: { [number]: number } },
				v: { [Types.Tensor]: { [number]: number } },
			}
		)
			lr = state.lr or lr
			b1 = state.b1 or b1
			b2 = state.b2 or b2
			eps = state.eps or eps
			t = state.t or t
			m = state.m or m
			v = state.v or v
		end,
	}
	return self
end
