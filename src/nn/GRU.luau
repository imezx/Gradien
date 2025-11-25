--!native
--!optimize 2
local Linear = require("./Linear")
local Tensor = require("../Tensor")
local Tape = require("../autograd/Tape")
local Types = require("../Types")

return function(inF: number, hidF: number): Types.Module<Types.Tensor, Types.Tensor>
	local Wx = Linear(inF, 3 * hidF)
	local Wh = Linear(hidF, 3 * hidF)
	local self
	self = {
		forward = function(_, x: Types.Tensor, h: Types.Tensor): Types.Tensor
			task.desynchronize()
			local gx = Wx:forward(x) -- {3H,B}
			local gh = Wh:forward(h) -- {3H,B}
			local B = x._shape[2]
			local z = Tensor.zeros({ hidF, B }, gx._dtype, gx._requiresGrad or gh._requiresGrad)
			local r = Tensor.zeros({ hidF, B }, gx._dtype, gx._requiresGrad or gh._requiresGrad)
			local n = Tensor.zeros({ hidF, B }, gx._dtype, gx._requiresGrad or gh._requiresGrad)
			local y = Tensor.zeros({ hidF, B }, gx._dtype, gx._requiresGrad or gh._requiresGrad)
			for i = 1, hidF do
				local zi = i -- z rows
				local ri = hidF + i -- r rows
				local ni = 2 * hidF + i -- n rows
				for b = 1, B do
					local zb = (zi - 1) * B + b
					local rb = (ri - 1) * B + b
					local nb = (ni - 1) * B + b
					local zpre = gx._storage[zb] + gh._storage[zb]
					local rpre = gx._storage[rb] + gh._storage[rb]
					local zv = 1 / (1 + math.exp(-zpre))
					local rv = 1 / (1 + math.exp(-rpre))
					z._storage[zb] = zv
					r._storage[rb] = rv
					local npre = gx._storage[nb] + rv * gh._storage[nb]
					local nv = math.tanh(npre)
					n._storage[nb] = nv
					y._storage[(i - 1) * B + b] = (1 - zv) * nv + zv * h._storage[(i - 1) * B + b]
				end
			end
			task.synchronize()
			return Tape.makeNode(y, { gx, gh, h }, function(gy: Types.Tensor)
				local B = y._shape[2]
				if gx._requiresGrad then
					gx._grad = gx._grad or Tensor.zeros(gx._shape, gx._dtype, false)
				end
				if gh._requiresGrad then
					gh._grad = gh._grad or Tensor.zeros(gh._shape, gh._dtype, false)
				end
				if h._requiresGrad then
					h._grad = h._grad or Tensor.zeros(h._shape, h._dtype, false)
				end
				task.desynchronize()
				for i = 1, hidF do
					local zi = i
					local ri = hidF + i
					local ni = 2 * hidF + i
					for b = 1, B do
						local hb = (i - 1) * B + b
						local zb = (zi - 1) * B + b
						local rb = (ri - 1) * B + b
						local nb = (ni - 1) * B + b
						local zv = z._storage[zb]
						local rv = r._storage[rb]
						local nv = n._storage[nb]
						local gyv = gy._storage[hb]
						-- y = (1 - z) * n + z * h
						local d_n = gyv * (1 - zv)
						local d_z = gyv * (h._storage[hb] - nv)
						if h._requiresGrad then
							h._grad._storage[hb] += gyv * zv
						end
						-- n = tanh(gx_n + r * gh_n)
						local dnpre = d_n * (1 - nv * nv)
						local d_gx_n = dnpre
						local d_gh_n = dnpre * rv
						local d_r = dnpre * gh._storage[nb]
						-- z,r sigmoids
						local dpre_z = d_z * zv * (1 - zv)
						local dpre_r = d_r * rv * (1 - rv)
						-- accumulate to gx/gh slices
						if gx._requiresGrad then
							gx._grad._storage[zb] += dpre_z
							gx._grad._storage[rb] += dpre_r
							gx._grad._storage[nb] += d_gx_n
						end
						if gh._requiresGrad then
							gh._grad._storage[zb] += dpre_z
							gh._grad._storage[rb] += dpre_r
							gh._grad._storage[nb] += d_gh_n
						end
					end
				end
				task.synchronize()
			end)
		end,
		parameters = function(_): { Types.Tensor }
			local p = Wx:parameters()
			for _, t in Wh:parameters() do
				table.insert(p, t)
			end
			return p
		end,
	}
	return self
end
