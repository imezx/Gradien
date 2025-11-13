--!optimize 2
local Activations = require("./Activations")
local Types = require("../Types")

return function(): Types.Module<Types.Tensor, Types.Tensor>
	return {
		--@param x Types.Tensor @ shape {2H, B}
		--@return Types.Tensor @ shape {H, B}
		forward = function(_, x: Types.Tensor): Types.Tensor
			return Activations.SwiGLUSplit(x)
		end,
		--@return {}
		parameters = function(_): {}
			return {}
		end,
	}
end
