--!native
--!optimize 2
local Tensor = require("../../Tensor")
local Tape = require("../../autograd/Tape")
local Types = require("../../Types")
local Linear = require("../../nn/Linear")
local Activations = require("../../nn/Activations")

local function RBF(x: Types.Tensor, grid: Types.Tensor, h: number): Types.Tensor
	-- x: {B, In}
	-- grid: {G} (centers)
	-- out: {B, In, G}
	local B, In = x._shape[1], x._shape[2]
	local G = grid._shape[1]
	local out = Tensor.zeros({ B, In, G }, x._dtype, x._requiresGrad or grid._requiresGrad)
	local xs = x._storage
	local gs = grid._storage
	local os = out._storage
	task.desynchronize()
	for b = 1, B do
		for i = 1, In do
			local idx_x = (b - 1) * In + i
			local val = xs[idx_x]
			for g = 1, G do
				local c = gs[g]
				local dist = (val - c) / h
				local rbf = math.exp(-dist * dist)
				-- out: {B, In, G}
				-- index: (b-1)*In*G + (i-1)*G + g
				local idx_o = ((b - 1) * In + (i - 1)) * G + g
				os[idx_o] = rbf
			end
		end
	end
	task.synchronize()
	return Tape.makeNode(out, { x, grid }, function(gy)
		if x._requiresGrad then
			if not x._grad then
				x._grad = Tensor.zeros(x._shape, x._dtype, false)
			end
			task.desynchronize()
			local x_grad = x._grad :: Types.Tensor
			local dxs = x_grad._storage
			local gys = gy._storage
			for b = 1, B do
				for i = 1, In do
					local idx_x = (b - 1) * In + i
					local val = xs[idx_x]
					local acc = 0
					for g = 1, G do
						local idx_o = ((b - 1) * In + (i - 1)) * G + g
						local gy_val = gys[idx_o]
						local c = gs[g]
						local dist = (val - c) / h
						local rbf = os[idx_o] -- exp term
						-- d/dx exp(-((x-c)/h)^2) = exp(...) * -2((x-c)/h) * 1/h
						local grad_val = rbf * (-2 * dist / h)
						acc += gy_val * grad_val
					end
					dxs[idx_x] += acc
				end
			end
			task.synchronize()
		end
	end)
end

local function KANLayer(in_features: number, out_features: number, grid_size: number?, spline_order: number?)
	grid_size = grid_size or 5
	spline_order = spline_order or 3
	local h = 1.0 / (grid_size :: number)
	local base_linear = Linear(in_features, out_features)
	local spline_linear = Linear(in_features * (grid_size :: number), out_features)
	local grid = Tensor.fromArray({}, { grid_size :: number }, "f64", false)
	local step = 2.0 / ((grid_size :: number) - 1)
	for i = 1, grid_size :: number do
		grid._storage[i] = -1 + (i - 1) * step
	end
	return {
		forward = function(x: Types.Tensor)
			local input_shape = x._shape
			local rank = #input_shape
			local flattened_dim = 1
			for i = 1, rank - 1 do
				flattened_dim *= input_shape[i]
			end
			local x_flat = Tensor.reshape(x, { flattened_dim, in_features })
			local x_silu = Activations.SiLU(x_flat)
			local base_out = base_linear:forward(x_silu)
			local x_rbf = RBF(x_flat, grid, h) -- {N, In, G}
			local x_rbf_flat = Tensor.reshape(x_rbf, { flattened_dim, in_features * (grid_size :: number) })
			local spline_out = spline_linear:forward(x_rbf_flat)
			local out = Tensor.zeros(base_out._shape, base_out._dtype, true)
			task.desynchronize()
			local os, bs, ss = out._storage, base_out._storage, spline_out._storage
			for i = 1, #os do
				os[i] = bs[i] + ss[i]
			end
			task.synchronize()
			out = Tape.makeNode(out, { base_out, spline_out }, function(gy)
				local gys = gy._storage
				if base_out._requiresGrad then
					if not base_out._grad then
						base_out._grad = Tensor.zeros(base_out._shape)
					end
					local gbs = base_out._grad._storage
					task.desynchronize()
					for i = 1, #gys do
						gbs[i] += gys[i]
					end
					task.synchronize()
				end
				if spline_out._requiresGrad then
					if not spline_out._grad then
						spline_out._grad = Tensor.zeros(spline_out._shape)
					end
					local gss = spline_out._grad._storage
					task.desynchronize()
					for i = 1, #gys do
						gss[i] += gys[i]
					end
					task.synchronize()
				end
			end)
			local out_shape = table.clone(input_shape)
			out_shape[rank] = out_features
			return Tensor.reshape(out, out_shape)
		end,
		parameters = function(_)
			local p = {}
			for _, v in base_linear:parameters() do
				table.insert(p, v)
			end
			for _, v in spline_linear:parameters() do
				table.insert(p, v)
			end
			return p
		end,
	}
end

return KANLayer
