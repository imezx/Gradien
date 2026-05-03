--!native
--!optimize 2
local Tensor = require("../Tensor")
local Tape = require("../autograd/Tape")
local Types = require("../Types")

return function(X: Types.Tensor, KH: number, KW: number, stride: number): Types.Tensor
	local C, H, W, B = X._shape[1], X._shape[2], X._shape[3], X._shape[4]
	local Ho = math.floor((H - KH) / stride) + 1
	local Wo = math.floor((W - KW) / stride) + 1
	local Y = Tensor.zeros({ C, Ho, Wo, B }, X._requiresGrad)
	task.desynchronize()
	local inputStorage = X._storage
	local outputStorage = Y._storage
	local sizeScale = 1 / (KH * KW)
	local HW_B = H * W * B
	local W_B = W * B
	for c = 1, C do
		local c_offset = (c - 1) * HW_B
		local c_out_offset = (c - 1) * Ho * Wo * B
		for b = 1, B do
			local b_offset = b - 1 
			for i = 1, Ho do
				local i_start = (i - 1) * stride
				local i_out_offset = (i - 1) * Wo * B
				for j = 1, Wo do
					local j_start = (j - 1) * stride
					local sum = 0
					for ki = 0, KH - 1 do
						local row_offset = (i_start + ki) * W_B
						for kj = 0, KW - 1 do
							local col_val = j_start + kj
							-- xi = c_offset + row_offset + col_val * B + b_offset
							local xi = c_offset + row_offset + col_val * B + b_offset
							sum += inputStorage[xi + 1]
						end
					end
					-- yi = c_out_offset + i_out_offset + (j-1)*B + (b-1) + 1
					local yi = c_out_offset + i_out_offset + (j - 1) * B + b_offset + 1
					outputStorage[yi] = sum * sizeScale
				end
			end
		end
	end
	task.synchronize()
	return Tape.makeNode(Y, { X }, function(gy: Types.Tensor)
		if X._requiresGrad then
			X._grad = X._grad or Tensor.zeros(X._shape, false)
			task.desynchronize()
			local scale = 1 / (KH * KW)
			local xGrad = X._grad :: Types.Tensor
			local xGradStorage = xGrad._storage
			local gyStorage = gy._storage
			local HW_B = H * W * B
			local W_B = W * B
			local HoWoB = Ho * Wo * B
			local WoB = Wo * B
			for c = 1, C do
				local c_offset = (c - 1) * HW_B
				local c_out_offset = (c - 1) * HoWoB
				for b = 1, B do
					local b_offset = b - 1
					for i = 1, Ho do
						local i_start = (i - 1) * stride
						local i_out_offset = (i - 1) * WoB
						for j = 1, Wo do
							local yi = c_out_offset + i_out_offset + (j - 1) * B + b_offset + 1
							local gyv = gyStorage[yi] * scale
							local j_start = (j - 1) * stride
							for ki = 0, KH - 1 do
								local row_offset = (i_start + ki) * W_B
								for kj = 0, KW - 1 do
									local col_val = j_start + kj
									local xi = c_offset + row_offset + col_val * B + b_offset
									xGradStorage[xi + 1] += gyv
								end
							end
						end
					end
				end
			end
			task.synchronize()
		end
	end)
end
