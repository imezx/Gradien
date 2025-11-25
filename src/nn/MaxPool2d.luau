--!native
--!optimize 2
local MP = require("../ops/Pool")
local Types = require("../Types")
return function(KH: number, KW: number, stride: number)
	return {
		forward = function(_, x: Types.Tensor): Types.Tensor
			return MP(x, KH, KW, stride)
		end,
		parameters = function(_)
			return {}
		end,
		train = function(_) end,
	}
end
