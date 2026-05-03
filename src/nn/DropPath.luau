--!native
--!optimize 2
local Tensor = require("../Tensor")
local Tape = require("../autograd/Tape")
local Types = require("../Types")

return function(p: number?, training: boolean?, rng: Random?): Types.Module<Types.Tensor, Types.Tensor>
	local p = p or 0.2
	local rng = rng or Random.new()
	local keep = 1 - p
	return {
		_train = type(training) == "boolean" and training or true,
		train = function(self, mode)
			self._train = mode == nil and true or mode
		end,
		parameters = function(self)
			return {}
		end,
		forward = function(self, x: Types.Tensor): Types.Tensor
			if not self._train or p <= 0 then
				return x
			end
			local D, B = x._shape[1], x._shape[2]
			local out = Tensor.zeros(x._shape, x._requiresGrad)
			task.desynchronize()
			local smask = table.create(B, 0)
			for b = 1, B do
				smask[b] = (rng:NextNumber(0, 1) < keep) and 1 or 0
			end
			for d = 1, D do
				local base = (d - 1) * B
				for b = 1, B do
					local m = smask[b]
					out._storage[base + b] = x._storage[base + b] * m / keep
				end
			end
			task.synchronize()
			if not out._requiresGrad then
				return out
			end
			return Tape.makeNode(out, { x }, function(gy: Types.Tensor)
				if x._requiresGrad then
					x._grad = x._grad or Tensor.zeros(x._shape, false)
					task.desynchronize()
					for d = 1, D do
						local base = (d - 1) * B
						for b = 1, B do
							x._grad._storage[base + b] += gy._storage[base + b] * smask[b] / keep
						end
					end
					task.synchronize()
				end
			end)
		end,
	}
end
