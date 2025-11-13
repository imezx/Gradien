--!optimize 2
local Tensor = require("../Tensor")
local Tape = require("../autograd/Tape")
local Types = require("../Types")

--@param f Types.Module<Types.Tensor, Types.Tensor>
return function(f: Types.Module<Types.Tensor, Types.Tensor>): Types.Module<Types.Tensor, Types.Tensor>
	return {
		forward = function(_, x: Types.Tensor): Types.Tensor
			local y = f:forward(x)
			assert(y._shape[1] == x._shape[1] and y._shape[2] == x._shape[2], "Residual: shape mismatch")

			local requiresGrad = (y._requiresGrad or x._requiresGrad) == true

			task.desynchronize()
			local out = Tensor.zeros(y._shape, y._dtype, requiresGrad)
			for i = 1, #y._storage do
				out._storage[i] = y._storage[i] + x._storage[i]
			end
			task.synchronize()

			if not requiresGrad then
				return out
			end

			return Tape.makeNode(out, { y, x }, function(gy: Types.Tensor)
				if y._requiresGrad then
					y._grad = y._grad or Tensor.zeros(y._shape)
					task.desynchronize()
					for i = 1, #gy._storage do
						y._grad._storage[i] += gy._storage[i]
					end
					task.synchronize()
				end
				if x._requiresGrad then
					x._grad = x._grad or Tensor.zeros(x._shape)
					task.desynchronize()
					for i = 1, #gy._storage do
						x._grad._storage[i] += gy._storage[i]
					end
					task.synchronize()
				end
			end)
		end,
		parameters = function(self)
			return f:parameters()
		end,
	}
end
