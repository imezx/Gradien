--!optimize 2
local Tensor = require("../Tensor")
local Reduce = require("../ops/Reduce")
local Tape = require("../autograd/Tape")
local Types = require("../Types")

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
	local runningMean = Tensor.zeros({ 1, 1 })
	local runningVar = Tensor.fromArray({ 1 }, { 1, 1 })
	return {
		_train = true,
		train = function(self, mode: boolean)
			self._train = mode == nil and true or mode
		end,
		forward = function(self, x: Types.Tensor): Types.Tensor
			local D_, B = x._shape[1], x._shape[2]
			assert(D_ == D, "BN1d D mismatch")
			task.desynchronize()
			local mu = Reduce.meanCols(x)
			local xc = Tensor.zeros(x._shape, x._dtype, x._requiresGrad)
			for j = 1, B do
				local m = mu._storage[j]
				for i = 1, D do
					xc._storage[(i - 1) * B + j] = x._storage[(i - 1) * B + j] - m
				end
			end
			task.synchronize()
			local var = Reduce.meanCols((function()
				task.desynchronize()
				local sq = Tensor.zeros(x._shape)
				for k = 1, #xc._storage do
					sq._storage[k] = xc._storage[k] ^ 2
				end
				task.synchronize()
				return sq
			end)())
			task.desynchronize()
			if self._train then
				runningMean._storage[1] = (1 - (momentum :: number)) * runningMean._storage[1]
					+ (momentum :: number) * mu._storage[1]
				runningVar._storage[1] = (1 - (momentum :: number)) * runningVar._storage[1]
					+ (momentum :: number) * var._storage[1]
			end
			local inv = Tensor.zeros({ 1, B })
			for j = 1, B do
				inv._storage[j] = 1 / math.sqrt(var._storage[j] + (eps :: number))
			end
			local y = Tensor.zeros(x._shape, x._dtype, x._requiresGrad)
			for j = 1, B do
				for i = 1, D do
					local idx = (i - 1) * B + j
					y._storage[idx] = gamma._storage[i] * xc._storage[idx] * inv._storage[j] + beta._storage[i]
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
							acc += gy._storage[idx] * xc._storage[idx] * inv._storage[j]
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
					for j = 1, B do
						for i = 1, D do
							local idx = (i - 1) * B + j
							x._grad._storage[idx] += gy._storage[idx] * gamma._storage[i] * inv._storage[j]
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
