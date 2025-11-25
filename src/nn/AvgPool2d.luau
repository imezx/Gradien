--!native
--!optimize 2
local Avg = require("../ops/AvgPool2d")
local Types = require("../Types")
return function(KH: number, KW: number, stride: number)
	return {
		forward = function(_, x: Types.Tensor): Types.Tensor
			return Avg(x, KH, KW, stride)
		end,
		parameters = function(_)
			return {}
		end,
		train = function(_) end,
	}
end
