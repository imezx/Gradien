--!native
--!optimize 2
local Tensor = require("../Tensor")
local Tape = require("../autograd/Tape")
local Types = require("../Types")

-- MaxPool2d kernel KHxKW stride s; input {C,H,W,B}
--@param X Tensor
--@param KH number
--@param KW number
--@param stride number
--@return Tensor
return function(X: Types.Tensor, KH: number, KW: number, stride: number): Types.Tensor
	local C, H, W, B = X._shape[1], X._shape[2], X._shape[3], X._shape[4]
	local Ho = math.floor((H - KH) / stride) + 1
	local Wo = math.floor((W - KW) / stride) + 1
	local Y = Tensor.zeros({ C, Ho, Wo, B }, X._requiresGrad)
	task.desynchronize()
	local argmax = table.create(C * Ho * Wo * B, 0)
	local HW_B = H * W * B
	local W_B = W * B
	local Wo_B = Wo * B
	local HoWoB = Ho * Wo * B
	local xStorage = X._storage
	local yStorage = Y._storage
	for c = 1, C do
		local c_offset = (c - 1) * HW_B
		local c_out_offset = (c - 1) * HoWoB
		for b = 1, B do
			local b_offset = b - 1
			for i = 1, Ho do
				local i_start = (i - 1) * stride
				local i_out_offset = (i - 1) * Wo_B
				for j = 1, Wo do
					local j_start = (j - 1) * stride
					local best = -math.huge
					local bestXi = 0
					for ki = 0, KH - 1 do
						local row_offset = (i_start + ki) * W_B
						for kj = 0, KW - 1 do
							local col_part = (j_start + kj) * B
							local xi = c_offset + row_offset + col_part + b_offset
							local v = xStorage[xi + 1]
							if v > best then
								best = v
								bestXi = xi
							end
						end
					end
					local yi = c_out_offset + i_out_offset + (j - 1) * B + b_offset + 1
					yStorage[yi] = best
					argmax[yi] = bestXi
				end
			end
		end
	end
	task.synchronize()
	return Tape.makeNode(Y, { X }, function(gy: Types.Tensor)
		if X._requiresGrad then
			X._grad = X._grad or Tensor.zeros(X._shape, false)
			task.desynchronize()
			local xGrad = X._grad :: Types.Tensor
			local xGradStorage = xGrad._storage
			local gyStorage = gy._storage
			local n = #gyStorage
			for idx = 1, n do
				local xi = argmax[idx]
				xGradStorage[xi + 1] += gyStorage[idx]
			end
			task.synchronize()
		end
	end)
end
