--!native
--!optimize 2
local Tensor = require("../Tensor")
local Tape = require("../autograd/Tape")
local convop = require("../ops/Conv2d")
local Types = require("../Types")

-- {C_in,H,W,B} -> {C_out,H-KH+1,W-KW+1,B}, stride=1, pad=0 (for now), with bias
return function(C_in: number, C_out: number, KH: number, KW: number): Types.Module<Types.Tensor, Types.Tensor>
	local W = Tensor.zeros({ C_out, C_in, KH, KW }, "f64", true)
	local b = Tensor.zeros({ C_out, 1 }, "f64", true)
	local self = {}
	self.parameters = function(_)
		return { W, b }
	end
	self.train = function(_) end
	self.forward = function(_, x: Types.Tensor): Types.Tensor
		local out = convop(x, W)
		-- add bias
		task.desynchronize()
		local F, Ho, Wo, B = out._shape[1], out._shape[2], out._shape[3], out._shape[4]
		for f = 1, F do
			local bf = b._storage[f]
			for i = 1, Ho do
				for j = 1, Wo do
					for bb = 1, B do
						local yi = (((f - 1) * Ho + (i - 1)) * Wo + (j - 1)) * B + bb
						out._storage[yi] += bf
					end
				end
			end
		end
		task.synchronize()
		local old_backprop = out._backprop
		local parents = out._parents and table.clone(out._parents) or {}
		table.insert(parents, b)
		if out._requiresGrad or b._requiresGrad then
			return Tape.makeNode(out, parents, function(gy: Types.Tensor)
				if b._requiresGrad then
					b._grad = b._grad or Tensor.zeros(b._shape, b._dtype, false)
					task.desynchronize()
					for f = 1, F do
						local acc = 0
						for i = 1, Ho do
							for j = 1, Wo do
								for bb = 1, B do
									local yi = (((f - 1) * Ho + (i - 1)) * Wo + (j - 1)) * B + bb
									acc += gy._storage[yi]
								end
							end
						end
						b._grad._storage[f] += acc
					end
					task.synchronize()
				end
				if old_backprop then
					old_backprop(gy)
				end
			end)
		end
		return out
	end
	return self
end
