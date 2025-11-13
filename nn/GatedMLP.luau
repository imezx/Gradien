--!optimize 2
local Linear = require("./Linear")
local A = require("./Activations")
local Types = require("../Types")

--@class GatedMLP
--@field fc_in any -- Linear(in, 2H)
--@field fc_out any -- Linear(H, out)
local GatedMLP = {}
GatedMLP.__index = GatedMLP

-- Construct a GatedMLP block.
--@param inDim number @ input width
--@param hidden number @ hidden width H (internal proj is 2H)
--@param outDim number @ output width
--@return GatedMLP
function GatedMLP.new(inDim: number, hidden: number, outDim: number)
	local self = setmetatable({}, GatedMLP)
	self.fc_in = Linear(inDim, 2 * hidden)
	self.fc_out = Linear(hidden, outDim)
	return self
end

-- Forward pass: y = W2 * SwiGLUSplit(W1 * x)
--@param x Types.Tensor @ shape {inDim, B}
--@return Types.Tensor @ shape {outDim, B}
function GatedMLP:forward(x: Types.Tensor): Types.Tensor
	local twoH = self.fc_in:forward(x) -- {2H, B}
	local h = A.SwiGLUSplit(twoH) -- {H, B}
	return self.fc_out:forward(h) -- {O, B}
end

--@return {Types.Tensor}
function GatedMLP:parameters(): { Types.Tensor }
	local p, a, b = {}, self.fc_in:parameters(), self.fc_out:parameters()
	for i = 1, #a do
		table.insert(p, a[i])
	end
	for i = 1, #b do
		table.insert(p, b[i])
	end
	return p
end

return GatedMLP.new
