--!optimize 2
local Types = require("../Types")

--@param params {Tensor}
--@param lr number
--@param betas {number}? @default {0.965, 0.99}
--@param rho number? @default 0.04
--@param weightDecay number? @default 0.0
return function(params: { Types.Tensor }, lr: number, betas: { number }?, rho: number?, weightDecay: number?)
	local b1 = (betas and betas[1]) or 0.965
	local b2 = (betas and betas[2]) or 0.99
	local rho_val = rho or 0.04
	local wd = weightDecay or 0.0
	local m: { [Types.Tensor]: { number } } = {}
	local h: { [Types.Tensor]: { number } } = {}
	local step = 0
	return {
		step = function(_)
			step += 1
			task.desynchronize()
			for _, p in params do
				if p._grad then
					local ps, gs = p._storage, p._grad._storage
					local ps_len = #ps
					if not m[p] then
						m[p] = table.create(ps_len, 0)
						h[p] = table.create(ps_len, 0)
					end
					local mi = m[p]
					local hi = h[p]
					for i = 1, ps_len do
						local g = gs[i]
						if wd ~= 0 then
							ps[i] -= lr * wd * ps[i]
						end
						local new_hi, new_mi = b2 * hi[i] + (1 - b2) * (g * g), b1 * mi[i] + (1 - b1) * g
						hi[i] = new_hi
						mi[i] = new_mi
						local hessian = math.max(math.sqrt(new_hi), rho_val)
						ps[i] -= lr * (new_mi / hessian)
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
		getState = function(): {
			lr: number,
			b1: number,
			b2: number,
			rho: number,
			step: number,
			m: { [Types.Tensor]: { number } },
			h: { [Types.Tensor]: { number } },
		}
			return { lr = lr, b1 = b1, b2 = b2, rho = rho_val, step = step, m = m, h = h }
		end,
		setState = function(
			_,
			state: {
				lr: number,
				b1: number,
				b2: number,
				rho: number,
				step: number,
				m: { [Types.Tensor]: { number } },
				h: { [Types.Tensor]: { number } },
			}
		)
			lr = state.lr or lr
			b1 = state.b1 or b1
			b2 = state.b2 or b2
			rho_val = state.rho or rho_val
			step = state.step or step
			m = state.m or m
			h = state.h or h
		end,
	}
end
