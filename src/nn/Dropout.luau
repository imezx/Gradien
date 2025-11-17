--!optimize 2
local Tensor = require("../Tensor")
local Tape = require("../autograd/Tape")
local Types = require("../Types")

return function(p: number?, training: boolean?, rng: Random?): Types.Module<Types.Tensor, Types.Tensor>
	local rng = rng or Random.new()
	local drop = p or 0.2
	local keep = 1 - drop
	return {
		_train = type(training) == "boolean" and training or true,
		train = function(self, mode)
			self._train = mode == nil and true or mode
		end,
		parameters = function(self)
			return {}
		end,
		forward = function(self, x: Types.Tensor): Types.Tensor
			if not self._train or drop <= 0 then
				return x
			end
			task.desynchronize()
			local y = Tensor.zeros(x._shape, x._dtype, x._requiresGrad)
			local mask = table.create(#x._storage, 0)
			for i = 1, #x._storage do
				local m = (rng:NextNumber(0, 1) >= drop) and 1 or 0
				mask[i] = m
				y._storage[i] = x._storage[i] * m / keep
			end
			task.synchronize()
			return Tape.makeNode(y, { x }, function(gy: Types.Tensor)
				if x._requiresGrad then
					task.desynchronize()
					if not x._grad then
						x._grad = Tensor.zeros(x._shape)
					end
					for i = 1, #x._grad._storage do
						x._grad._storage[i] += gy._storage[i] * mask[i] / keep
					end
					task.synchronize()
				end
			end)
		end,
	}
end
