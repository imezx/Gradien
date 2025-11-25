--!native
--!optimize 2
local Tensor = require("../Tensor")
local Tape = require("../autograd/Tape")
local Types = require("../Types")
local Util = require("../Util")
local assert = Util.Assert

--@param C number
--@return table
return function(C: number): Types.Module<Types.Tensor, Types.Tensor>
	local gamma = Tensor.fromArray(table.create(C, 1), { C, 1 }, "f64", true)
	local beta = Tensor.zeros({ C, 1 }, "f64", true)
	local self
	self = {
		forward = function(_, x: Types.Tensor): Types.Tensor
			local Cx, H, W, B = x._shape[1], x._shape[2], x._shape[3], x._shape[4]
			assert(Cx == C, "BN2d C mismatch")
			task.desynchronize()
			local M = H * W * B
			local y = Tensor.zeros(
				{ C, H, W, B },
				x._dtype,
				(x._requiresGrad or gamma._requiresGrad or beta._requiresGrad) == true
			)
			local mu = table.create(C, 0)
			local inv = table.create(C, 0)
			for c = 1, C do
				local sum = 0
				for i = 1, H do
					for j = 1, W do
						for b = 1, B do
							local xi = (((c - 1) * H + (i - 1)) * W + (j - 1)) * B + b
							sum += x._storage[xi]
						end
					end
				end
				mu[c] = sum / M
			end
			local eps = 1e-5
			for c = 1, C do
				local v = 0
				for i = 1, H do
					for j = 1, W do
						for b = 1, B do
							local xi = (((c - 1) * H + (i - 1)) * W + (j - 1)) * B + b
							local d = x._storage[xi] - mu[c]
							v += d * d
						end
					end
				end
				inv[c] = 1 / math.sqrt(v / M + eps)
			end
			-- y = gamma * (x - mu) * inv + beta
			for c = 1, C do
				local g = gamma._storage[c]
				local bt = beta._storage[c]
				local muc = mu[c]
				local ic = inv[c]
				for i = 1, H do
					for j = 1, W do
						for b = 1, B do
							local xi = (((c - 1) * H + (i - 1)) * W + (j - 1)) * B + b
							y._storage[xi] = g * (x._storage[xi] - muc) * ic + bt
						end
					end
				end
			end
			task.synchronize()
			return Tape.makeNode(y, { x, gamma, beta }, function(gy: Types.Tensor)
				local M = H * W * B
				-- dBeta, dGamma
				if beta._requiresGrad or gamma._requiresGrad then
					task.desynchronize()
					if beta._requiresGrad then
						beta._grad = beta._grad or Tensor.zeros({ C, 1 }, beta._dtype, false)
						for c = 1, C do
							local acc = 0
							for i = 1, H do
								for j = 1, W do
									for b = 1, B do
										local yi = (((c - 1) * H + (i - 1)) * W + (j - 1)) * B + b
										acc += gy._storage[yi]
									end
								end
							end
							beta._grad._storage[c] += acc
						end
					end
					if gamma._requiresGrad then
						gamma._grad = gamma._grad or Tensor.zeros({ C, 1 }, gamma._dtype, false)
						for c = 1, C do
							local acc = 0
							local muc, ic = mu[c], inv[c]
							for i = 1, H do
								for j = 1, W do
									for b = 1, B do
										local xi = (((c - 1) * H + (i - 1)) * W + (j - 1)) * B + b
										local xhat = (x._storage[xi] - muc) * ic
										local yi = xi
										acc += gy._storage[yi] * xhat
									end
								end
							end
							gamma._grad._storage[c] += acc
						end
					end
					task.synchronize()
				end
				-- dX
				if x._requiresGrad then
					x._grad = x._grad or Tensor.zeros(x._shape, x._dtype, false)
					task.desynchronize()
					for c = 1, C do
						local muc, ic = mu[c], inv[c]
						local g = gamma._storage[c]
						local sumGy, sumGyXhat = 0, 0
						for i = 1, H do
							for j = 1, W do
								for b = 1, B do
									local xi = (((c - 1) * H + (i - 1)) * W + (j - 1)) * B + b
									local yi = xi
									local gyij = gy._storage[yi]
									sumGy += gyij
									sumGyXhat += gyij * ((x._storage[xi] - muc) * ic)
								end
							end
						end
						for i = 1, H do
							for j = 1, W do
								for b = 1, B do
									local xi = (((c - 1) * H + (i - 1)) * W + (j - 1)) * B + b
									local gyij = gy._storage[xi]
									local term = (M * gyij - sumGy - ((x._storage[xi] - muc) * ic) * sumGyXhat) / M
									x._grad._storage[xi] += g * ic * term
								end
							end
						end
					end
					task.synchronize()
				end
			end)
		end,
		parameters = function()
			return { gamma, beta }
		end,
	}
	return self
end
