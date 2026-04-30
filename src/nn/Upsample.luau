--!native
--!optimize 2
local Tensor = require("../Tensor")
local Tape = require("../autograd/Tape")
local Types = require("../Types")

return function(scaleFactor: number, H: number?, W: number?)
	local sf = math.max(1, math.floor(scaleFactor))
	return {
		forward = function(_, x: Types.Tensor): Types.Tensor
			local C = x._shape[1]
			local L = x._shape[2]
			local B = x._shape[3] or 1

			if H and W then
				-- 2D spatial: treat L as H*W, upsample H and W independently
				assert(L == H * W, "Upsample: H*W must equal x._shape[2]")
				local Hout = H * sf
				local Wout = W * sf
				local Lout = Hout * Wout
				local out = Tensor.zeros({ C, Lout, B }, x._dtype, x._requiresGrad)
				task.desynchronize()
				for c = 1, C do
					for h = 1, Hout do
						local hs = math.floor((h - 1) / sf) + 1
						for w = 1, Wout do
							local ws = math.floor((w - 1) / sf) + 1
							local srcL = (hs - 1) * W + ws
							local dstL = (h - 1) * Wout + w
							for b = 1, B do
								out._storage[(c - 1) * Lout * B + (dstL - 1) * B + b] =
									x._storage[(c - 1) * L * B + (srcL - 1) * B + b]
							end
						end
					end
				end
				task.synchronize()

				if not x._requiresGrad then
					return out
				end
				return Tape.makeNode(out, { x }, function(gy: Types.Tensor)
					if not x._requiresGrad then
						return
					end
					x._grad = x._grad or Tensor.zeros(x._shape)
					local gx = (x._grad :: Types.Tensor)._storage
					local gys = gy._storage
					task.desynchronize()
					for c = 1, C do
						for h = 1, Hout do
							local hs = math.floor((h - 1) / sf) + 1
							for w = 1, Wout do
								local ws = math.floor((w - 1) / sf) + 1
								local srcL = (hs - 1) * W + ws
								local dstL = (h - 1) * Wout + w
								for b = 1, B do
									gx[(c - 1) * L * B + (srcL - 1) * B + b] += gys[(c - 1) * Lout * B + (dstL - 1) * B + b]
								end
							end
						end
					end
					task.synchronize()
				end)
			else
				-- 1D: upsample sequence axis
				local Lout = L * sf
				local out = Tensor.zeros({ C, Lout, B }, x._dtype, x._requiresGrad)
				task.desynchronize()
				for c = 1, C do
					for l = 1, Lout do
						local ls = math.floor((l - 1) / sf) + 1
						for b = 1, B do
							out._storage[(c - 1) * Lout * B + (l - 1) * B + b] =
								x._storage[(c - 1) * L * B + (ls - 1) * B + b]
						end
					end
				end
				task.synchronize()

				if not x._requiresGrad then
					return out
				end
				return Tape.makeNode(out, { x }, function(gy: Types.Tensor)
					if not x._requiresGrad then
						return
					end
					x._grad = x._grad or Tensor.zeros(x._shape)
					local gx = (x._grad :: Types.Tensor)._storage
					local gys = gy._storage
					task.desynchronize()
					for c = 1, C do
						for l = 1, Lout do
							local ls = math.floor((l - 1) / sf) + 1
							for b = 1, B do
								gx[(c - 1) * L * B + (ls - 1) * B + b] += gys[(c - 1) * Lout * B + (l - 1) * B + b]
							end
						end
					end
					task.synchronize()
				end)
			end
		end,
		parameters = function(_): { Types.Tensor }
			return {}
		end,
		train = function(_) end,
	}
end
