--!native
--!optimize 2
local Tensor = require("../Tensor")
local Tape = require("../autograd/Tape")
local Types = require("../Types")

--@param f Types.Module<Types.Tensor, Types.Tensor> - The main sub-network (e.g., LayerNorm -> GatedMLP)
--@param attn Types.Module<Types.Tensor, Types.Tensor> - The attention sub-network (e.g., LayerNorm -> Linear -> Sigmoid)
return function(
	f: Types.Module<Types.Tensor, Types.Tensor>,
	attn: Types.Module<Types.Tensor, Types.Tensor>
): Types.Module<Types.Tensor, Types.Tensor>
	return {
		forward = function(_, x: Types.Tensor): Types.Tensor
			local y = f:forward(x)
			local a = attn:forward(x)

			assert(y._shape[1] == x._shape[1] and y._shape[2] == x._shape[2], "AttnResBlock: main path shape mismatch")
			assert(
				a._shape[1] == x._shape[1] and a._shape[2] == x._shape[2],
				"AttnResBlock: attention path shape mismatch"
			)

			local requiresGrad = (y._requiresGrad or x._requiresGrad or a._requiresGrad) == true

			task.desynchronize()
			-- out = x + (a * y)
			local out = Tensor.zeros(y._shape, requiresGrad)
			for i = 1, #y._storage do
				out._storage[i] = x._storage[i] + (a._storage[i] * y._storage[i])
			end
			task.synchronize()

			if not requiresGrad then
				return out
			end

			return Tape.makeNode(out, { y, a, x }, function(gy: Types.Tensor)
				-- Gradients for: out = x + a * y
				-- dy = gy * a
				-- da = gy * y
				-- dx = gy

				if y._requiresGrad then
					y._grad = y._grad or Tensor.zeros(y._shape)
					task.desynchronize()
					for i = 1, #gy._storage do
						y._grad._storage[i] += gy._storage[i] * a._storage[i]
					end
					task.synchronize()
				end

				if a._requiresGrad then
					a._grad = a._grad or Tensor.zeros(a._shape)
					task.desynchronize()
					for i = 1, #gy._storage do
						a._grad._storage[i] += gy._storage[i] * y._storage[i]
					end
					task.synchronize()
				end

				if x._requiresGrad then
					x._grad = x._grad or Tensor.zeros(x._shape)
					task.desynchronize()
					for i = 1, #gy._storage do
						(x._grad :: any)._storage[i] += gy._storage[i]
					end
					task.synchronize()
				end
			end)
		end,
		parameters = function(self): { Tensor.Tensor }
			local params = {}
			for _, p in f:parameters() do
				table.insert(params, p)
			end
			for _, p in attn:parameters() do
				table.insert(params, p)
			end
			return params
		end,
	}
end
