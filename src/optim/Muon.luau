--!native
--!optimize 2
local Types = require("../Types")
local Tensor = require("../Tensor")
local BLAS = require("../ops/BLAS")

-- X_{k+1} = 1.5 * X_k - 0.5 * X_k @ X_k^T @ X_k
local function newtonSchulz(G: Types.Tensor, steps: number): Types.Tensor
	local X = Tensor.detach(G)
	local _rows, _cols = X._shape[1], X._shape[2]
	task.desynchronize()
	local sumSq = 0
	local xs, xs_len = X._storage, #X._storage
	for i = 1, xs_len do
		sumSq += xs[i] * xs[i]
	end
	local frob = math.sqrt(sumSq)
	if frob > 1e-6 then
		for i = 1, xs_len do
			xs[i] /= frob
		end
	end
	task.synchronize()
	for k = 1, steps do
		-- A = X @ X.T
		-- B = A @ X
		-- X = 1.5 * X - 0.5 * B
		local XT = Tensor.transpose(X)
		local A = BLAS.matmul(X, XT)
		local B = BLAS.matmul(A, X)
		local bs = B._storage
		local xs_curr = X._storage
		task.desynchronize()
		for i = 1, #xs_curr do
			xs_curr[i] = 1.5 * xs_curr[i] - 0.5 * bs[i]
		end
		task.synchronize()
	end
	return X
end

return function(params: { Types.Tensor }, lr: number, momentum: number?, nesterov: boolean?, nsSteps: number?)
	momentum = momentum or 0.95
	nsSteps = nsSteps or 3
	local state: { [Types.Tensor]: Types.Tensor } = {}
	local step = 0
	local self = {
		step = function(_)
			step += 1
			for _, p in params do
				if p._grad then
					local g = p._grad
					local m = state[p]
					if not m then
						m = Tensor.zeros(p._shape, p._dtype, false)
						state[p] = m
					end
					task.desynchronize()
					local ms, gs = m._storage, g._storage
					for i = 1, #ms do
						ms[i] = (momentum :: number) * ms[i] + gs[i]
					end
					task.synchronize()
					local rank = #p._shape
					if rank == 2 then
						local orth = newtonSchulz(m, nsSteps :: number)
						local sumW = 0
						local ps, ps_len = p._storage, #p._storage
						local os = orth._storage
						task.desynchronize()
						for i = 1, ps_len do
							sumW += ps[i] * ps[i]
						end
						local rmsW = math.sqrt(sumW / ps_len)
						local scale = math.max(1, rmsW)
						for i = 1, ps_len do
							ps[i] -= lr * scale * os[i]
						end
						task.synchronize()
					else
						task.desynchronize()
						local ps = p._storage
						for i = 1, #ps do
							ps[i] -= lr * ms[i]
						end
						task.synchronize()
					end
				end
			end
		end,
		zeroGrad = function(_)
			for _, p in params do
				p._grad = nil
			end
		end,
		getState = function()
			return {
				lr = lr,
				momentum = momentum,
				nsSteps = nsSteps,
				step = step,
			}
		end,
		setLr = function() end,
	}
	return self
end
