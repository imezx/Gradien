--!optimize 2
local Tensor = require("../Tensor")
local Tape = require("../autograd/Tape")
local Types = require("../Types")

-- Affine layer y = W x + b
-- @param inF  number  (fan-in)
-- @param outF number  (fan-out)
-- @param init (fanIn:number, fanOut:number) -> number
-- @return Types.Module
return function(
	inF: number,
	outF: number,
	init: ((number, number) -> number)?
): Types.Module<Types.Tensor, Types.Tensor>
	-- W: {outF, inF}, b: {outF,1}
	local W = Tensor.zeros({ outF, inF }, "f64", true)
	local b = Tensor.zeros({ outF, 1 }, "f64", true)
	if type(init) == "function" then
		task.desynchronize()
		local store = W._storage
		local idx = 1
		for i = 1, outF do
			for _ = 1, inF do
				store[idx] = init(inF, outF)
				idx += 1
			end
		end
		task.synchronize()
	end
	return {
		W = W,
		b = b,
		-- x: {inF,B}  -> y: {outF,B}
		forward = function(_, x: Types.Tensor): Types.Tensor
			task.desynchronize()
			local out = Tape.matmul(W, x)
			local B = x._shape[2]
			for i = 1, outF do
				local bi = b._storage[i]
				for j = 1, B do
					out._storage[(i - 1) * B + j] += bi
				end
			end
			task.synchronize()
			return Tape.makeNode(out, { W, b, x }, function(gy: Types.Tensor)
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
