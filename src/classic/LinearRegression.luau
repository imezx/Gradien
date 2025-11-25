--!native
--!optimize 2
local Tape = require("../autograd/Tape")
local Losses = require("../nn/Losses")
local Linear = require("../nn/Linear")
local SGD = require("../optim/SGD")

return function(inF: number)
	local model = Linear(inF, 1)
	local self
	self = {
		fit = function(_, X, Y, epochs: number, lr: number)
			local opt = SGD(model:parameters(), lr or 1e-2)
			for _ = 1, epochs do
				opt:zeroGrad()
				local pred = model:forward(X)
				local _, grad = Losses.mse_backward(pred, Y)
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
