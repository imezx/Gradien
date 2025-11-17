--!optimize 2
local Tensor = require("../Tensor")
local Tape = require("../autograd/Tape")
local Reduce = require("../ops/Reduce")
local Types = require("../Types")

-- LayerNorm across features for each column (shape {D,B}).
return function(D: number, eps: number?): Types.Module<Types.Tensor, Types.Tensor>
	eps = eps or 1e-5
	local gamma = Tensor.fromArray(table.create(D, 1), { D, 1 }, "f64", true)
	local beta = Tensor.zeros({ D, 1 }, "f64", true)
	local self
	self = {
		forward = function(_, x: Types.Tensor): Types.Tensor
			local D_, B = x._shape[1], x._shape[2]
			assert(D_ == D, "LayerNorm D mismatch")
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
			local norm = Tensor.zeros(x._shape, x._dtype, x._requiresGrad)
			for j = 1, B do
				local inv = 1 / math.sqrt(var._storage[j] + (eps :: number))
				for i = 1, D do
					norm._storage[(i - 1) * B + j] = xc._storage[(i - 1) * B + j] * inv
				end
			end
			local y = Tensor.zeros(x._shape, x._dtype, x._requiresGrad)
			for j = 1, B do
				for i = 1, D do
					local idx = (i - 1) * B + j
					y._storage[idx] = gamma._storage[i] * norm._storage[idx] + beta._storage[i]
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
							acc += gy._storage[idx] * norm._storage[idx]
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
					local inv = table.create(B, 0)
					for j = 1, B do
						inv[j] = 1 / math.sqrt(var._storage[j] + (eps :: number))
					end
					for j = 1, B do
						local invJ = inv[j]
						local sumDxhat = 0
						local sumDxhatXhat = 0
						for i = 1, D do
							local idx = (i - 1) * B + j
							local dxhat = gy._storage[idx] * gamma._storage[i]
							sumDxhat += dxhat
							sumDxhatXhat += dxhat * norm._storage[idx]
						end
						for i = 1, D do
							local idx = (i - 1) * B + j
							local dxhat = gy._storage[idx] * gamma._storage[i]
							local term = dxhat - sumDxhat / D - norm._storage[idx] * sumDxhatXhat / D
							xGrad._storage[idx] += term * invJ
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
	return self
end
