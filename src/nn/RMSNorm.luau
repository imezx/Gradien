--!native
--!optimize 2
local Tensor = require("../Tensor")
local Tape = require("../autograd/Tape")
local Types = require("../Types")

local function forward(self: { weight: Types.Tensor, eps: number }, x: Types.Tensor): Types.Tensor
	task.desynchronize()
	local D, B = x._shape[1], x._shape[2]
	local y = Tensor.zeros({ D, B }, x._requiresGrad)
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
		if x._requiresGrad or w._requiresGrad then
			task.desynchronize()
			if w._requiresGrad and not w._grad then
				w._grad = Tensor.zeros(w._shape)
			end
			if x._requiresGrad and not x._grad then
				x._grad = Tensor.zeros(x._shape)
			end
			for j = 1, B do
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
				for i = 1, D do
					local idx = (i - 1) * B + j
					local xij = x._storage[idx]
					local gij = gy._storage[idx]
					if w._requiresGrad then
						w._grad._storage[i] += (xij / r) * gij
					end
					if x._requiresGrad and x._grad then
						x._grad._storage[idx] += (w._storage[i] / r) * gij - (xij / (D * r3)) * inner
					end
				end
			end
			task.synchronize()
		end
	end)
end

local function parameters(self: { weight: Types.Tensor, eps: number }): { Types.Tensor }
	return { self.weight }
end

local function new(D: number, eps: number?)
	local weight = Tensor.zeros({ D, 1 }, true)
	task.desynchronize()
	for i = 1, #weight._storage do
		weight._storage[i] = 1
	end
	task.synchronize()
	return {
		weight = weight,
		eps = eps or 1e-5,
		forward = forward,
		parameters = parameters,
	}
end

return new
