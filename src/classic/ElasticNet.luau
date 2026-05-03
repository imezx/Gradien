--!native
--!optimize 2
local Tensor = require("../Tensor")
local Tape = require("../autograd/Tape")
local Losses = require("../nn/Losses")
local Linear = require("../nn/Linear")
local Adam = require("../optim/Adam")

-- ElasticNet: MSE + lambda1 * ||W||_1 + lambda2 * ||W||_2^2
-- lambda1 controls L1 sparsity (Lasso), lambda2 controls L2 shrinkage (Ridge).
-- Setting lambda1=0 recovers Ridge; setting lambda2=0 recovers Lasso.
return function(inF: number, lambda1: number?, lambda2: number?)
	local l1 = lambda1 or 1e-3
	local l2 = lambda2 or 1e-3
	local model = Linear(inF, 1)

	local self
	self = {
		fit = function(_, X, Y, epochs: number?, lr: number?, batchSize: number?)
			local epochs_ = epochs or 100
			local lr_ = lr or 1e-3
			local D, N = X._shape[1], X._shape[2]
			local bs = math.min(batchSize or N, N)
			local opt = Adam(model:parameters(), lr_)

			for _ = 1, epochs_ do
				-- mini-batch loop
				local i = 1
				while i <= N do
					local last = math.min(i + bs - 1, N)
					local B = last - i + 1

					-- slice columns [i, last] from X and Y
					local XB = Tensor.zeros({ D, B }, X._requiresGrad)
					local YB = Tensor.zeros({ 1, B }, false)
					task.desynchronize()
					for d = 1, D do
						for b = 1, B do
							XB._storage[(d - 1) * B + b] = X._storage[(d - 1) * N + (i + b - 1)]
						end
					end
					for b = 1, B do
						YB._storage[b] = Y._storage[i + b - 1]
					end
					task.synchronize()

					opt:zeroGrad()
					local pred = model:forward(XB)
					local _, grad = Losses.mse_backward(pred, YB)

					-- inject elastic-net penalty into W._grad before backward
					local W = model.W
					if not W._grad then
						W._grad = Tensor.zeros(W._shape)
					end
					task.desynchronize()
					local ws, gs = W._storage, W._grad._storage
					for k = 1, #ws do
						local w = ws[k]
						-- L1: subgradient sign(w); L2: 2*lambda2*w
						local sign = (w > 0) and 1 or (w < 0 and -1 or 0)
						gs[k] += l1 * sign + 2 * l2 * w
					end
					task.synchronize()

					Tape.backwardFrom(pred, grad)
					opt:step()
					i = last + 1
				end
			end
		end,

		predict = function(_, X)
			return model:forward(X)
		end,

		parameters = function(_)
			return model:parameters()
		end,
	}
	return self
end
