--!optimize 2
local Tensor = require("../Tensor")
local Tape = require("../autograd/Tape")
local Losses = require("../nn/Losses")
local Linear = require("../nn/Linear")
local SGD = require("../optim/SGD")

return function(inF: number, wd: number)
	wd = wd or 1e-2
	local model = Linear(inF, 1)
	local self
	self = {
		fit = function(_, X, Y, epochs: number, lr: number)
			local opt = SGD(model:parameters(), lr or 1e-2, 0.9, true)
			for _ = 1, epochs do
				opt:zeroGrad()
				local pred = model:forward(X)
				local _, grad = Losses.mse_backward(pred, Y)
				local W = model.W
				if not W._grad then
					W._grad = Tensor.zeros(W._shape)
				end
				task.desynchronize()
				for i = 1, #W._storage do
					W._grad._storage[i] += 2 * wd * W._storage[i]
				end
				task.synchronize()
				Tape.backwardFrom(pred, grad)
				opt:step()
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
