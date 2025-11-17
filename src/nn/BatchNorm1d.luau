--!optimize 2
local Tensor = require("../Tensor")
local Tape = require("../autograd/Tape")
local Types = require("../Types")
local Util = require("../Util")
local assert = Util.Assert

-- BatchNorm1d (shape {D,B})
--@param D number
--@param eps number?
--@param momentum number?
--@return table
return function(D: number, eps: number?, momentum: number?): Types.Module<Types.Tensor, Types.Tensor>
	eps = eps or 1e-5
	momentum = momentum or 0.1
	local gamma = Tensor.fromArray(table.create(D, 1), { D, 1 }, "f64", true)
	local beta = Tensor.zeros({ D, 1 }, "f64", true)
	local runningMean = Tensor.zeros({ D, 1 })
	local runningVar = Tensor.fromArray(table.create(D, 1), { D, 1 })
	return {
		_train = true,
		train = function(self, mode: boolean)
			self._train = mode == nil and true or mode
		end,
		forward = function(self, x: Types.Tensor): Types.Tensor
			local D_, B = x._shape[1], x._shape[2]
			assert(D_ == D, "BN1d D mismatch")
			local xc = Tensor.zeros(x._shape, x._dtype, x._requiresGrad)
			local meanUsed: Types.Tensor
			local varUsed: Types.Tensor
			if self._train then
				meanUsed = Tensor.zeros({ D, 1 })
				varUsed = Tensor.zeros({ D, 1 })
				task.desynchronize()
				for i = 1, D do
					local sum = 0
					local base = (i - 1) * B
					for j = 1, B do
						sum += x._storage[base + j]
					end
					local mean = sum / B
					meanUsed._storage[i] = mean
					local vsum = 0
					for j = 1, B do
						local idx = base + j
						local centered = x._storage[idx] - mean
						xc._storage[idx] = centered
						vsum += centered * centered
					end
					varUsed._storage[i] = vsum / B
				end
				task.synchronize()
				task.desynchronize()
				for i = 1, D do
					runningMean._storage[i] = (1 - (momentum :: number)) * runningMean._storage[i]
						+ (momentum :: number) * meanUsed._storage[i]
					runningVar._storage[i] = (1 - (momentum :: number)) * runningVar._storage[i]
						+ (momentum :: number) * varUsed._storage[i]
				end
				task.synchronize()
			else
				meanUsed = runningMean
				varUsed = runningVar
				task.desynchronize()
				for i = 1, D do
					local base = (i - 1) * B
					local mean = meanUsed._storage[i]
					for j = 1, B do
						local idx = base + j
						xc._storage[idx] = x._storage[idx] - mean
					end
				end
				task.synchronize()
			end
			local inv: Types.Tensor = Tensor.zeros({ D, 1 })
			local y = Tensor.zeros(x._shape, x._dtype, x._requiresGrad)
			task.desynchronize()
			for i = 1, D do
				local invVal = 1 / math.sqrt(varUsed._storage[i] + (eps :: number))
				inv._storage[i] = invVal
				local base = (i - 1) * B
				local g = gamma._storage[i]
				local b = beta._storage[i]
				for j = 1, B do
					local idx = base + j
					y._storage[idx] = g * xc._storage[idx] * invVal + b
				end
			end
			task.synchronize()
			return Tape.makeNode(y, { x, gamma, beta }, function(gy: Types.Tensor)
				if gamma._requiresGrad then
					task.desynchronize()
					if not gamma._grad then
						gamma._grad = Tensor.zeros({ D, 1 })
					end
					for i = 1, D do
						local acc = 0
						for j = 1, B do
							local idx = (i - 1) * B + j
							acc += gy._storage[idx] * xc._storage[idx] * inv._storage[i]
						end
						gamma._grad._storage[i] += acc
					end
					task.synchronize()
				end
				if beta._requiresGrad then
					task.desynchronize()
					if not beta._grad then
						beta._grad = Tensor.zeros({ D, 1 })
					end
					for i = 1, D do
						local acc = 0
						for j = 1, B do
							local idx = (i - 1) * B + j
							acc += gy._storage[idx]
						end
						beta._grad._storage[i] += acc
					end
					task.synchronize()
				end
				if x._requiresGrad then
					task.desynchronize()
					if not x._grad then
						x._grad = Tensor.zeros(x._shape)
					end
					local xGrad = x._grad :: Types.Tensor
					local sumGy = table.create(D, 0)
					local sumGyXhat = table.create(D, 0)
					for i = 1, D do
						local base = (i - 1) * B
						local invI = inv._storage[i]
						local accGy = 0
						local accGyXhat = 0
						for j = 1, B do
							local idx = base + j
							local gyVal = gy._storage[idx]
							accGy += gyVal
							accGyXhat += gyVal * (xc._storage[idx] * invI)
						end
						sumGy[i] = accGy
						sumGyXhat[i] = accGyXhat
					end
					local M = B
					for i = 1, D do
						local base = (i - 1) * B
						local g = gamma._storage[i]
						local invI = inv._storage[i]
						local sumGyI = sumGy[i]
						local sumGyXhatI = sumGyXhat[i]
						for j = 1, B do
							local idx = base + j
							local gyVal = gy._storage[idx]
							local xhat = xc._storage[idx] * invI
							local term = (M * gyVal - sumGyI - xhat * sumGyXhatI) / M
							xGrad._storage[idx] += g * invI * term
						end
					end
					task.synchronize()
				end
			end)
		end,
		parameters = function(_)
			return { gamma, beta }
		end,
	}
end
