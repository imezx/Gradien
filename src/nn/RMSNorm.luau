--!optimize 2
local Tensor = require("../Tensor")
local Tape = require("../autograd/Tape")
local Types = require("../Types")

--@class RMSNorm
--@field weight any -- {D,1}
--@field eps number
local RMSNorm = {}
RMSNorm.__index = RMSNorm

--@param D number  -- feature dim
--@param eps number? -- default 1e-5
--@return RMSNorm
function RMSNorm.new(D: number, eps: number?)
	local self = setmetatable({}, RMSNorm)
	self.eps = eps or 1e-5
	self.weight = Tensor.zeros({ D, 1 }, "f32", true)
	task.desynchronize()
	for i = 1, #self.weight._storage do
		self.weight._storage[i] = 1
	end
	task.synchronize()
	return self
end

-- y = x * weight / sqrt(mean(x^2) + eps) (per column)
function RMSNorm:forward(x: Types.Tensor): Types.Tensor
	task.desynchronize()
	local D, B = x._shape[1], x._shape[2]
	local y = Tensor.zeros({ D, B }, x._dtype, x._requiresGrad)
	local w = self.weight

	for j = 1, B do
		local sqsum = 0.0
		for i = 1, D do
			local idx = (i - 1) * B + j
			local v = x._storage[idx]
			sqsum += v * v
		end
		local r = math.sqrt(sqsum / D + self.eps) -- denom
		for i = 1, D do
			local idx = (i - 1) * B + j
			y._storage[idx] = x._storage[idx] * w._storage[i] / r
		end
	end
	task.synchronize()

	return Tape.makeNode(y, { x, w }, function(gy: Types.Tensor)
		-- grads: dw_i = sum_j gy_ij * x_ij / r_j
		--  dx_kj = (w_k/r_j)*gy_kj - (x_kj/(D*r_j^3)) * sum_i (w_i*x_ij*gy_ij)
		if x._requiresGrad or w._requiresGrad then
			task.desynchronize()
			if w._requiresGrad and not w._grad then
				w._grad = Tensor.zeros(w._shape)
			end
			if x._requiresGrad and not x._grad then
				x._grad = Tensor.zeros(x._shape)
			end
			for j = 1, B do
				-- recompute r_j and the inner sum
				local sqsum = 0.0
				for i = 1, D do
					local idx = (i - 1) * B + j
					local v = x._storage[idx]
					sqsum += v * v
				end
				local r = math.sqrt(sqsum / D + self.eps)
				local r3 = r * r * r

				local inner = 0.0
				for i = 1, D do
					local idx = (i - 1) * B + j
					inner += w._storage[i] * x._storage[idx] * gy._storage[idx]
				end
				-- grads per feature
				for i = 1, D do
					local idx = (i - 1) * B + j
					local xij = x._storage[idx]
					local gij = gy._storage[idx]
					if w._requiresGrad then
						w._grad._storage[i] += (xij / r) * gij
					end
					if x._requiresGrad then
						if x._grad then
							x._grad._storage[idx] += (w._storage[i] / r) * gij - (xij / (D * r3)) * inner
						end
					end
				end
			end
			task.synchronize()
		end
	end)
end

function RMSNorm:parameters()
	return { self.weight }
end

return RMSNorm.new
