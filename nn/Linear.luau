--!optimize 2
local Tensor = require("../Tensor")
local Tape = require("../autograd/Tape")
local Types = require("../Types")

-- Affine layer y = W x + b
--@param inF number
--@param outF number
--@return table
return function(inF: number, outF: number): Types.Module<Types.Tensor, Types.Tensor>
	local W = Tensor.zeros({ outF, inF }, "f64", true)
	local b = Tensor.zeros({ outF, 1 }, "f64", true)
	return {
		W = W,
		b = b,
		-- Forward pass.
		--@param x any @Tensor {inF,B}
		--@return any @Tensor {outF,B}
		forward = function(_, x: Types.Tensor): Types.Tensor
			task.desynchronize()
			local out = Tape.matmul(W, x)
			-- add bias per column
			local B = x._shape[2]
			for i = 1, outF do
				local bi = b._storage[i]
				for j = 1, B do
					out._storage[(i - 1) * B + j] += bi
				end
			end
			task.synchronize()
			return Tape.makeNode(out, { W, b, x }, function(gy: Types.Tensor)
				-- W and b grads are handled via matmul above + bias add split
				if b._requiresGrad then
					task.desynchronize()
					if not b._grad then
						b._grad = Tensor.zeros({ outF, 1 })
					end
					for i = 1, outF do
						local acc = 0
						for j = 1, B do
							acc += gy._storage[(i - 1) * B + j]
						end
						b._grad._storage[i] += acc
					end
					task.synchronize()
				end
			end)
		end,
		parameters = function(_)
			return { W, b }
		end,
	}
end
