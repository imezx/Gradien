--!optimize 2
local Linear = require("./Linear")
local Tensor = require("../Tensor")
local Tape = require("../autograd/Tape")
local Types = require("../Types")

return function(inF: number, hidF: number): Types.Module<Types.Tensor, Types.Tensor>
	local Wx = Linear(inF, hidF)
	local Wh = Linear(hidF, hidF)
	local self
	self = {
		forward = function(_, x: Types.Tensor, h: Types.Tensor): Types.Tensor
			-- gates
			task.desynchronize()
			local gx = Wx:forward(x) -- {H,B}
			local gh = Wh:forward(h) -- {H,B}
			local B = gx._shape[2]
			local y = Tensor.zeros({ hidF, B }, gx._dtype, (gx._requiresGrad or gh._requiresGrad) == true)
			for i = 1, hidF do
				for b = 1, B do
					local idx = (i - 1) * B + b
					local pre = gx._storage[idx] + gh._storage[idx]
					y._storage[idx] = math.tanh(pre)
				end
			end
			task.synchronize()
			return Tape.makeNode(y, { gx, gh }, function(gy: Types.Tensor)
				local B = y._shape[2]
				if gx._requiresGrad or gh._requiresGrad then
					task.desynchronize()
					if gx._requiresGrad then
						gx._grad = gx._grad or Tensor.zeros(gx._shape, gx._dtype, false)
					end
					if gh._requiresGrad then
						gh._grad = gh._grad or Tensor.zeros(gh._shape, gh._dtype, false)
					end
					for i = 1, hidF do
						for b = 1, B do
							local idx = (i - 1) * B + b
							local dpre = (1 - y._storage[idx] * y._storage[idx]) * gy._storage[idx]
							if gx._requiresGrad then
								gx._grad._storage[idx] += dpre
							end
							if gh._requiresGrad then
								gh._grad._storage[idx] += dpre
							end
						end
					end
					task.synchronize()
				end
			end)
		end,
		parameters = function(_)
			local p = Wx:parameters()
			for _, t in Wh:parameters() do
				table.insert(p, t)
			end
			return p
		end,
	}
	return self
end
