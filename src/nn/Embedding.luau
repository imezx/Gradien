--!native
--!optimize 2
local Tensor = require("../Tensor")
local Tape = require("../autograd/Tape")
local Types = require("../Types")

return function(numEmbeds: number, dim: number): Types.Module<Types.Tensor, Types.Tensor>
	local W = Tensor.zeros({ dim, numEmbeds }, "f64", true)
	return {
		forward = function(_, idx: { number }): Types.Tensor
			task.desynchronize()
			local B = #idx
			local Y = Tensor.zeros({ dim, B }, "f64", W._requiresGrad)
			for j = 1, B do
				local k = math.clamp(idx[j], 1, numEmbeds)
				for i = 1, dim do
					Y._storage[(i - 1) * B + j] = W._storage[(i - 1) * numEmbeds + k]
				end
			end
			task.synchronize()
			return Tape.makeNode(Y, { W }, function(gY: Types.Tensor)
				if W._requiresGrad then
					task.desynchronize()
					if not W._grad then
						W._grad = Tensor.zeros({ dim, numEmbeds })
					end
					for j = 1, B do
						local k = math.clamp(idx[j], 1, numEmbeds)
						for i = 1, dim do
							W._grad._storage[(i - 1) * numEmbeds + k] += gY._storage[(i - 1) * B + j]
						end
					end
					task.synchronize()
				end
			end)
		end,
		parameters = function(_)
			return { W }
		end,
	}
end
