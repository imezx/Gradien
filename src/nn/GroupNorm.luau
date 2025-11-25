--!native
--!optimize 2
local Tensor = require("../Tensor")
local Tape = require("../autograd/Tape")
local Types = require("../Types")
local Util = require("../Util")
local assert = Util.Assert

return function(num_groups: number, num_channels: number, eps: number?): Types.Module<Types.Tensor, Types.Tensor>
	eps = eps or 1e-5
	assert(num_channels % num_groups == 0, "GroupNorm: num_channels must be divisible by num_groups")

	local gamma = Tensor.ones({ num_channels, 1 }, "f64", true)
	local beta = Tensor.zeros({ num_channels, 1 }, "f64", true)

	local self
	self = {
		forward = function(_, x: Types.Tensor): Types.Tensor
			local C = x._shape[1]
			assert(C == num_channels, "GroupNorm: channel mismatch")
			local shape = x._shape
			local rank = #shape
			local B = shape[rank]
			local spatial_size = 1
			for i = 2, rank - 1 do
				spatial_size *= shape[i]
			end
			local G = num_groups
			local C_per_G = C / G
			local group_size = C_per_G * spatial_size
			local xs = x._storage
			task.desynchronize()
			local stats = table.create(G * B * 2, 0)
			local stride_C = Util.sizeFromShape(shape) / C
			for c = 1, C do
				local g = math.ceil(c / C_per_G)
				local g_idx = (g - 1) * B
				local c_base = (c - 1) * stride_C
				for i = 0, stride_C - 1 do
					local b = (i % B) + 1
					local val = xs[c_base + i + 1]
					stats[g_idx + b] += val
				end
			end
			local inv_size = 1 / group_size
			for k = 1, G * B do
				stats[k] *= inv_size
			end
			for c = 1, C do
				local g = math.ceil(c / C_per_G)
				local g_idx = (g - 1) * B
				local c_base = (c - 1) * stride_C
				for i = 0, stride_C - 1 do
					local b = (i % B) + 1
					local val = xs[c_base + i + 1]
					local m = stats[g_idx + b]
					stats[G * B + g_idx + b] += (val - m) ^ 2
				end
			end
			for k = G * B + 1, G * B * 2 do
				stats[k] = math.sqrt(stats[k] * inv_size + eps)
			end
			local y = Tensor.zeros(shape, x._dtype, x._requiresGrad)
			local ys = y._storage
			local gs = gamma._storage
			local bs = beta._storage
			for c = 1, C do
				local g = math.ceil(c / C_per_G)
				local g_idx = (g - 1) * B
				local c_base = (c - 1) * stride_C
				local gam = gs[c]
				local bet = bs[c]
				for i = 0, stride_C - 1 do
					local b = (i % B) + 1
					local val = xs[c_base + i + 1]
					local m = stats[g_idx + b]
					local s = stats[G * B + g_idx + b]
					-- y = gamma * (x - mean) / std + beta
					ys[c_base + i + 1] = gam * (val - m) / s + bet
				end
			end
			task.synchronize()
			return Tape.makeNode(y, { x, gamma, beta }, function(gy: Types.Tensor)
				if x._requiresGrad then
					if not x._grad then
						x._grad = Tensor.zeros(shape, x._dtype, false)
					end
				end
				if gamma._requiresGrad and not gamma._grad then
					gamma._grad = Tensor.zeros({ C, 1 }, "f64", false)
				end
				if beta._requiresGrad and not beta._grad then
					beta._grad = Tensor.zeros({ C, 1 }, "f64", false)
				end
				task.desynchronize()
				local gys = gy._storage
				local dxs = x._requiresGrad and (x._grad :: Types.Tensor)._storage
				local dgs = gamma._requiresGrad and gamma._grad._storage
				local dbs = beta._requiresGrad and beta._grad._storage
				local back_stats = table.create(G * B * 2, 0)
				for c = 1, C do
					local g = math.ceil(c / C_per_G)
					local g_idx = (g - 1) * B
					local c_base = (c - 1) * stride_C
					local gam = gs[c]
					for i = 0, stride_C - 1 do
						local b = (i % B) + 1
						local idx = c_base + i + 1
						local gy_val = gys[idx]
						local val = xs[idx]
						local m = stats[g_idx + b]
						local s = stats[G * B + g_idx + b]
						local x_hat = (val - m) / s
						if dgs then
							dgs[c] += gy_val * x_hat
						end
						if dbs then
							dbs[c] += gy_val
						end
						local dy_gamma = gy_val * gam
						back_stats[g_idx + b] += dy_gamma
						back_stats[G * B + g_idx + b] += dy_gamma * x_hat
					end
				end
				if dxs then
					local N = group_size
					local dxs_nonil = dxs :: { number }
					for c = 1, C do
						local g = math.ceil(c / C_per_G)
						local g_idx = (g - 1) * B
						local c_base = (c - 1) * stride_C
						local gam = gs[c]
						for i = 0, stride_C - 1 do
							local b = (i % B) + 1
							local idx = c_base + i + 1
							local m = stats[g_idx + b]
							local s = stats[G * B + g_idx + b]
							local val = xs[idx]
							local x_hat = (val - m) / s
							local sum_dy_gamma = back_stats[g_idx + b]
							local sum_dy_gamma_xhat = back_stats[G * B + g_idx + b]
							local dy = gys[idx]
							-- dL/dx = (1/s) * ( dy*gamma - sum_dy_gamma/N - x_hat * sum_dy_gamma_xhat/N )
							local term = (dy * gam) - (sum_dy_gamma / N) - (x_hat * sum_dy_gamma_xhat / N)
							dxs_nonil[idx] += term / s
						end
					end
				end
				task.synchronize()
			end)
		end,
		parameters = function(_)
			return { gamma, beta }
		end,
	}
	return self
end
