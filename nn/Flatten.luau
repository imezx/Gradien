--!optimize 2
local Tensor = require("../Tensor")
local Tape = require("../autograd/Tape")
local Types = require("../Types")

-- Flattens everything but the last dimension (batch) for NCHW+B -> {C*H*W,B}
return function(): Types.Module<Types.Tensor, Types.Tensor>
	return {
		forward = function(_, x: Types.Tensor): Types.Tensor
			local shape = x._shape
			if #shape == 2 then
				return x
			end
			if #shape == 4 then
				local C, H, W, B = shape[1], shape[2], shape[3], shape[4]
				local out = Tensor.zeros({ C * H * W, B }, x._dtype, x._requiresGrad)
				task.desynchronize()
				for c = 1, C do
					for i = 1, H do
						for j = 1, W do
							for b = 1, B do
								local xi = (((c - 1) * H + (i - 1)) * W + (j - 1)) * B + b
								local yi = ((((c - 1) * H + (i - 1)) * W + (j - 1)) * B) + b
								out._storage[yi] = x._storage[xi + 1]
							end
						end
					end
				end
				task.synchronize()
				if not out._requiresGrad then
					return out
				end
				return Tape.makeNode(out, { x }, function(gy: Types.Tensor)
					if x._requiresGrad then
						x._grad = x._grad or Tensor.zeros(x._shape, x._dtype, false)
						task.desynchronize()
						for c = 1, C do
							for i = 1, H do
								for j = 1, W do
									for b = 1, B do
										local xi = (((c - 1) * H + (i - 1)) * W + (j - 1)) * B + b
										local yi = ((((c - 1) * H + (i - 1)) * W + (j - 1)) * B) + b
										x._grad._storage[xi + 1] += gy._storage[yi]
									end
								end
							end
						end
						task.synchronize()
					end
				end)
			else
				error("Flatten: unsupported tensor rank " .. tostring(#shape))
			end
		end,
		parameters = function(_)
			return {}
		end,
		train = function(_) end,
	}
end
