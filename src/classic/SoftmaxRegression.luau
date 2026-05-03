--!native
--!optimize 2
local Linear = require("../nn/Linear")
local Losses = require("../nn/Losses")
local Tape = require("../autograd/Tape")
local Tensor = require("../Tensor")
local Types = require("../Types")
local Adam = require("../optim/Adam")
local Util = require("../Util")

local assert = Util.Assert

local function slice(
	N: number,
	X: Types.Tensor,
	D: number,
	y: { number },
	startIdx: number,
	size: number
): (Types.Tensor, { number })
	local B = math.min(size, N - startIdx + 1)
	local XB = Tensor.zeros({ D, B }, X._requiresGrad)
	local yB = table.create(B, 1)
	task.desynchronize()
	for j = 1, B do
		local srcj = startIdx + j - 1
		for i = 1, D do
			XB._storage[(i - 1) * B + j] = X._storage[(i - 1) * N + srcj]
		end
		yB[j] = y[srcj]
	end
	task.synchronize()
	return XB, yB
end

local function forward(self, X: Types.Tensor): Types.Tensor
	return self.fc:forward(X)
end

local function parameters(self): { Types.Tensor }
	return self.fc:parameters()
end

local function fit(self, X: Types.Tensor, y: { number }, lr: number, epochs: number, batch: number, smoothing: number)
	lr, epochs, batch = lr or 1e-2, epochs or 5, batch or 64
	smoothing = smoothing or 0
	local D, N = X._shape[1], X._shape[2]
	assert(#y == N, "y length must equal N")

	task.desynchronize()
	local opt = Adam(parameters(self), lr)
	for _ = 1, epochs do
		for s = 1, N, batch do
			local XB, yB = slice(N, X, D, y, s, batch)
			opt:zeroGrad()
			local logits = forward(self, XB)
			local _, dlogits = Losses.cross_entropy_backward(logits, yB, smoothing)
			Tape.backwardFrom(logits, dlogits)
			opt:step()
		end
	end
	task.synchronize()
end

local function new(inDim: number, classes: number): Types.Module<Types.Tensor, Types.Tensor>
	return {
		fc = Linear(inDim, classes),
		forward = forward,
		parameters = parameters,
		fit = fit,
	}
end

return new
