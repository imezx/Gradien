--!optimize 2
local Linear = require("../nn/Linear")
local Losses = require("../nn/Losses")
local Tape = require("../autograd/Tape")
local Tensor = require("../Tensor")
local Adam = require("../optim/Adam")
local Util = require("../Util")

local assert = Util.Assert

--@class SoftmaxRegression
--@field fc any
local SoftmaxRegression = {}
SoftmaxRegression.__index = SoftmaxRegression

local function slice(N, X, D, y, startIdx: number, size: number)
	local B = math.min(size, N - startIdx + 1)
	local XB = Tensor.zeros({ D, B }, X._dtype, X._requiresGrad)
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

--@param inDim number -- D
--@param classes number -- C
--@return SoftmaxRegression
function SoftmaxRegression.new(inDim: number, classes: number)
	local self = setmetatable({}, SoftmaxRegression)
	self.fc = Linear(inDim, classes)
	return self
end

--@param X Tensor {D,B}
--@return logits {C,B}
function SoftmaxRegression:forward(X)
	return self.fc:forward(X)
end

function SoftmaxRegression:parameters()
	return self.fc:parameters()
end

--@param X Tensor -- {D,N}
--@param y {number} -- indices 1..C, length N
--@param lr number? -- default 1e-2
--@param epochs integer? -- default 5
--@param batch integer? -- default 64
--@param smoothing number? -- default 0
function SoftmaxRegression:fit(X, y, lr, epochs, batch, smoothing)
	lr, epochs, batch = lr or 1e-2, epochs or 5, batch or 64
	smoothing = smoothing or 0
	local D, N = X._shape[1], X._shape[2]
	assert(#y == N, "y length must equal N")
	
	task.desynchronize()
	local opt = Adam(self:parameters(), lr)
	for e = 1, epochs do
		for s = 1, N, batch do
			local XB, yB = slice(N, X, D, y, s, batch)
			opt:zeroGrad()
			local logits = self:forward(XB)
			local _, dlogits = Losses.cross_entropy_backward(logits, yB, smoothing)
			Tape.backwardFrom(logits, dlogits)
			opt:step()
		end
	end
	task.synchronize()
end

return SoftmaxRegression.new
