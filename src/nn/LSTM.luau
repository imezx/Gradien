--!native
--!optimize 2
local Linear = require("./Linear")
local Tensor = require("../Tensor")
local Tape = require("../autograd/Tape")
local Types = require("../Types")

return function(inF: number, hidF: number): Types.Module<Types.Tensor, Types.Tensor>
	local Wx = Linear(inF, 4 * hidF)
	local Wh = Linear(hidF, 4 * hidF)
	local self
	self = {
		forward = function(_, x: Types.Tensor, h: Types.Tensor, c: Types.Tensor): Types.Tensor
			task.desynchronize()
			local B = x._shape[2]
			local gx = Wx:forward(x) -- {4H,B}
			local gh = Wh:forward(h) -- {4H,B}
			local i = Tensor.zeros({ hidF, B }, gx._dtype, gx._requiresGrad or gh._requiresGrad)
			local f = Tensor.zeros({ hidF, B }, gx._dtype, gx._requiresGrad or gh._requiresGrad)
			local g = Tensor.zeros({ hidF, B }, gx._dtype, gx._requiresGrad or gh._requiresGrad)
			local o = Tensor.zeros({ hidF, B }, gx._dtype, gx._requiresGrad or gh._requiresGrad)
			local cNew = Tensor.zeros(
				{ hidF, B },
				gx._dtype,
				(gx._requiresGrad or gh._requiresGrad or (c and c._requiresGrad)) == true
			)
			local hNew = Tensor.zeros(
				{ hidF, B },
				gx._dtype,
				(gx._requiresGrad or gh._requiresGrad or (c and c._requiresGrad)) == true
			)
			for j = 1, hidF do
				local ii = j
				local fi = hidF + j
				local gi = 2 * hidF + j
				local oi = 3 * hidF + j
				for b = 1, B do
					local ib = (ii - 1) * B + b
					local fb = (fi - 1) * B + b
					local gb = (gi - 1) * B + b
					local ob = (oi - 1) * B + b
					local pre_i = gx._storage[ib] + gh._storage[ib]
					local pre_f = gx._storage[fb] + gh._storage[fb]
					local pre_g = gx._storage[gb] + gh._storage[gb]
					local pre_o = gx._storage[ob] + gh._storage[ob]
					local iv = 1 / (1 + math.exp(-pre_i))
					local fv = 1 / (1 + math.exp(-pre_f))
					local gv = math.tanh(pre_g)
					local ov = 1 / (1 + math.exp(-pre_o))
					i._storage[ib] = iv
					f._storage[fb] = fv
					g._storage[gb] = gv
					o._storage[ob] = ov
					local cb = (j - 1) * B + b
					local cprev = c._storage[cb]
					local cnow = fv * cprev + iv * gv
					cNew._storage[cb] = cnow
					hNew._storage[cb] = ov * math.tanh(cnow)
				end
			end
			task.synchronize()
			local hNode = Tape.makeNode(hNew, { gx, gh, c }, function(gyh)
				local B = hNew._shape[2]
				if gx._requiresGrad then
					gx._grad = gx._grad or Tensor.zeros(gx._shape, gx._dtype, false)
				end
				if gh._requiresGrad then
					gh._grad = gh._grad or Tensor.zeros(gh._shape, gh._dtype, false)
				end
				if c._requiresGrad then
					c._grad = c._grad or Tensor.zeros(c._shape, c._dtype, false)
				end
				task.desynchronize()
				for j = 1, hidF do
					local ii = j
					local fi = hidF + j
					local gi = 2 * hidF + j
					local oi = 3 * hidF + j
					for b = 1, B do
						local ib = (ii - 1) * B + b
						local fb = (fi - 1) * B + b
						local gb = (gi - 1) * B + b
						local ob = (oi - 1) * B + b
						local cb = (j - 1) * B + b
						local cnow = cNew._storage[cb]
						local tanh_c = math.tanh(cnow)
						local gyhv = gyh._storage[cb]
						-- h' = o * tanh(c')
						local d_o = gyhv * tanh_c
						local d_c_from_h = gyhv * o._storage[ob] * (1 - tanh_c * tanh_c)
						local dpre_o = d_o * o._storage[ob] * (1 - o._storage[ob])
						if gx._requiresGrad then
							gx._grad._storage[ob] += dpre_o
						end
						if gh._requiresGrad then
							gh._grad._storage[ob] += dpre_o
						end
						if c._requiresGrad then
							local dct = d_c_from_h
							-- c' = f*c + i*g
							local d_i = dct * g._storage[gb]
							local d_f = dct * c._storage[cb]
							local d_g = dct * i._storage[ib]

							local dpre_i = d_i * i._storage[ib] * (1 - i._storage[ib])
							local dpre_f = d_f * f._storage[fb] * (1 - f._storage[fb])
							local dpre_g = d_g * (1 - g._storage[gb] * g._storage[gb])
							if gx._requiresGrad then
								gx._grad._storage[ib] += dpre_i
								gx._grad._storage[fb] += dpre_f
								gx._grad._storage[gb] += dpre_g
							end
							if gh._requiresGrad then
								gh._grad._storage[ib] += dpre_i
								gh._grad._storage[fb] += dpre_f
								gh._grad._storage[gb] += dpre_g
							end
							c._grad._storage[cb] += dct * f._storage[fb]
						end
					end
				end
				task.synchronize()
			end)

			local cNode = Tape.makeNode(cNew, { gx, gh, c }, function(gyc)
				local B = cNew._shape[2]
				if gx._requiresGrad then
					gx._grad = gx._grad or Tensor.zeros(gx._shape, gx._dtype, false)
				end
				if gh._requiresGrad then
					gh._grad = gh._grad or Tensor.zeros(gh._shape, gh._dtype, false)
				end
				if c._requiresGrad then
					c._grad = c._grad or Tensor.zeros(c._shape, c._dtype, false)
				end
				task.desynchronize()
				for j = 1, hidF do
					local ii = j
					local fi = hidF + j
					local gi = 2 * hidF + j
					for b = 1, B do
						local ib = (ii - 1) * B + b
						local fb = (fi - 1) * B + b
						local gb = (gi - 1) * B + b
						local cb = (j - 1) * B + b
						local dct = gyc._storage[cb]
						-- c' = f*c + i*g
						local d_i = dct * g._storage[gb]
						local d_f = dct * c._storage[cb]
						local d_g = dct * i._storage[ib]

						local dpre_i = d_i * i._storage[ib] * (1 - i._storage[ib])
						local dpre_f = d_f * f._storage[fb] * (1 - f._storage[fb])
						local dpre_g = d_g * (1 - g._storage[gb] * g._storage[gb])
						if gx._requiresGrad then
							gx._grad._storage[ib] += dpre_i
							gx._grad._storage[fb] += dpre_f
							gx._grad._storage[gb] += dpre_g
						end
						if gh._requiresGrad then
							gh._grad._storage[ib] += dpre_i
							gh._grad._storage[fb] += dpre_f
							gh._grad._storage[gb] += dpre_g
						end
						if c._requiresGrad then
							c._grad._storage[cb] += dct * f._storage[fb]
						end
					end
				end
				task.synchronize()
			end)
			return hNode, cNode
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
