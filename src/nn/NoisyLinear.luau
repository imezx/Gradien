local Tensor = require("../Tensor")
local Base = require("./Linear")
local Tape = require("../autograd/Tape")
local Types = require("../Types")

return function(inF: number, outF: number, sigma: number?, rng: Random?): Types.Module<Types.Tensor, Types.Tensor>
	sigma = sigma or 0.5
	rng = rng or Random.new()
	local base = Base(inF, outF)
	local bsigma = Tensor.zeros({ outF, 1 }, true)
	local function resample()
		task.desynchronize()
		for i = 1, #bsigma._storage do
			bsigma._storage[i] = rng:NextNumber(-sigma, sigma)
		end
		task.synchronize()
	end
	return {
		forward = function(_, x: Types.Tensor): Types.Tensor
			local y = base:forward(x)
			task.desynchronize()
			local B = x._shape[2]
			for i = 1, outF do
				local bi = bsigma._storage[i]
				for j = 1, B do
					y._storage[(i - 1) * B + j] += bi
				end
			end
			task.synchronize()
			return Tape.makeNode(y, { base.b, bsigma }, function(gy: Types.Tensor)
				local Bsz = y._shape[2]
				if base.b._requiresGrad then
					base.b._grad = base.b._grad or Tensor.zeros(base.b._shape)
					task.desynchronize()
					for i = 1, outF do
						local acc = 0
						for j = 1, Bsz do
							acc += gy._storage[(i - 1) * Bsz + j]
						end
						base.b._grad._storage[i] += acc
					end
					task.synchronize()
				end
				if bsigma._requiresGrad then
					bsigma._grad = bsigma._grad or Tensor.zeros(bsigma._shape)
					task.desynchronize()
					for i = 1, outF do
						local acc = 0
						for j = 1, Bsz do
							acc += gy._storage[(i - 1) * Bsz + j]
						end
						bsigma._grad._storage[i] += acc
					end
					task.synchronize()
				end
			end)
		end,
		parameters = function(_)
			local p = base:parameters()
			table.insert(p, bsigma)
			return p
		end,
		resample = resample,
	}
end
