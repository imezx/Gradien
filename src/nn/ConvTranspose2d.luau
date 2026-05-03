--!native
--!optimize 2
local Tensor = require("../Tensor")
local Tape = require("../autograd/Tape")
local convT = require("../ops/ConvTranspose2d")
local Types = require("../Types")

--{C_in,H,W,B} -> {C_out, H+KH-1, W+KW-1, B}
return function(C_in: number, C_out: number, KH: number, KW: number): Types.Module<Types.Tensor, Types.Tensor>
	local W = Tensor.zeros({ C_in, C_out, KH, KW }, true)
	local b = Tensor.zeros({ C_out, 1 }, true)
	local self = {}
	self.parameters = function(_)
		return { W, b }
	end
	self.train = function(_) end
	self.forward = function(_, x: Types.Tensor): Types.Tensor
		local out = convT(x, W)
		task.desynchronize()
		local F, Ho, Wo, B = out._shape[1], out._shape[2], out._shape[3], out._shape[4]
		local os = out._storage
		local bs = b._storage
		local WoB = Wo * B
		for f = 1, F do
			local bf = bs[f]
			local f_offset = (f - 1) * Ho * Wo * B
			for i = 1, Ho do
				local i_offset = (i - 1) * WoB
				for j = 1, Wo do
					local j_offset = (j - 1) * B
					for bb = 1, B do
						local idx = f_offset + i_offset + j_offset + bb
						os[idx] += bf
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
					b._grad = b._grad or Tensor.zeros(b._shape, false)
					task.desynchronize()
					local gbs = b._grad._storage
					local gys = gy._storage
					for f = 1, F do
						local acc = 0
						local f_offset = (f - 1) * Ho * Wo * B
						for i = 1, Ho do
							local i_offset = (i - 1) * WoB
							for j = 1, Wo do
								local j_offset = (j - 1) * B
								for bb = 1, B do
									local idx = f_offset + i_offset + j_offset + bb
									acc += gys[idx]
								end
							end
						end
						gbs[f] += acc
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
