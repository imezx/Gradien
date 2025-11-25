--!native
--!optimize 2
local SoftmaxOps = require("../ops/Softmax")
local Types = require("../Types")

local S = {}

function S.forward(logits: Types.Tensor): Types.Tensor
	return SoftmaxOps.forward(logits)
end

return S
