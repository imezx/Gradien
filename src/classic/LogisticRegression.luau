--!native
--!optimize 2
local Tensor = require("../Tensor")
local Tape = require("../autograd/Tape")
local Losses = require("../nn/Losses")
local Linear = require("../nn/Linear")
local Adam = require("../optim/Adam")

return function(inF: number)
	local model = Linear(inF, 1)
	local self
	self = {
		fit = function(_, X, targets: { number }, epochs: number, lr: number)
			local opt = Adam(model:parameters(), lr or 1e-3)
			for _ = 1, epochs do
				opt:zeroGrad()
				local logits = model:forward(X)
				local _, grad = Losses.bceWithLogits_backward(logits, targets)
				Tape.backwardFrom(logits, grad)
				opt:step()
			end
		end,
		predictProba = function(_, X)
			local logits = model:forward(X)
			local out = Tensor.zeros(logits._shape)
			task.desynchronize()
			for i = 1, #logits._storage do
				out._storage[i] = 1 / (1 + math.exp(-logits._storage[i]))
			end
			task.synchronize()
			return out
		end,
		predict = function(self, X, threshold: number?)
			threshold = threshold or 0.5
			local p = self:predictProba(X)
			local r = table.create(p._shape[2], 0)
			task.desynchronize()
			for j = 1, #r do
				r[j] = (p._storage[j] >= threshold) and 1 or 0
			end
			task.synchronize()
			return r
		end,
		parameters = function(_)
			return model:parameters()
		end,
	}
	return self
end
