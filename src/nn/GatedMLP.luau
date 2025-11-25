--!native
--!optimize 2
local Linear = require("./Linear")
local A = require("./Activations")
local Types = require("../Types")

--@class GatedMLP
--@field fc_in any -- Linear(in, 2H)
--@field fc_out any -- Linear(H, out)

local function forward(self, x: Types.Tensor): Types.Tensor
	local twoH = self.fc_in:forward(x) -- {2H, B}
	local h = A.SwiGLUSplit(twoH) -- {H, B}
	return self.fc_out:forward(h) -- {O, B}
end

local function parameters(self): { Types.Tensor }
	local p, a, b = {}, self.fc_in:parameters(), self.fc_out:parameters()
	for i = 1, #a do
		p[#p + 1] = a[i]
	end
	for i = 1, #b do
		p[#p + 1] = b[i]
	end
	return p
end

local function new(inDim: number, hidden: number, outDim: number)
	return {
		fc_in = Linear(inDim, 2 * hidden),
		fc_out = Linear(hidden, outDim),
		forward = forward,
		parameters = parameters,
	}
end

return new
