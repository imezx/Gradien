--!native
--!optimize 2
local Tensor = require("../Tensor")
local Tape = require("../autograd/Tape")
local Types = require("../Types")

local function ensureGrad(t: Types.Tensor): Types.Tensor?
	if not t._grad then
		t._grad = Tensor.zeros(t._shape, t._dtype, false)
	end
	return t._grad
end

return function(
	inF: number,
	outF: number,
	init: ((number, number) -> number)?
): Types.Module<Types.Tensor, Types.Tensor>
	local W = Tensor.zeros({ outF, inF }, "f64", true)
	local b = Tensor.zeros({ outF, 1 }, "f64", true)
	if type(init) == "function" then
		task.desynchronize()
		local store = W._storage
		local idx = 1
		for i = 1, outF do
			for _ = 1, inF do
				store[idx] = init(inF, outF)
				idx += 1
			end
		end
		task.synchronize()
	end
	return {
		W = W,
		b = b,
		forward = function(_, x: Types.Tensor): Types.Tensor
			-- A(W): {outF, inF}, B(x): {inF, Batch} -> C: {outF, Batch}
			local M, K = outF, inF
			local K2, N = x._shape[1], x._shape[2]
			if K ~= K2 then
				error("Linear: input shape mismatch")
			end
			local out = Tensor.zeros({ M, N }, W._dtype, W._requiresGrad or b._requiresGrad or x._requiresGrad)
			task.desynchronize()
			local W_store = W._storage
			local x_store = x._storage
			local b_store = b._storage
			local out_store = out._storage
			for i = 1, M do
				local bias_val = b_store[i]
				local w_offset = (i - 1) * K
				local out_offset = (i - 1) * N
				for j = 1, N do
					local acc = 0
					for k = 1, K do
						-- W index: w_offset + k
						-- x index: (k-1)*N + j
						acc += W_store[w_offset + k] * x_store[(k - 1) * N + j]
					end
					out_store[out_offset + j] = acc + bias_val -- Fuse bias add
				end
			end
			task.synchronize()
			return Tape.makeNode(out, { W, b, x }, function(gy: Types.Tensor)
				-- dW = gy @ x^T
				-- db = sum(gy, dim=1)
				-- dx = W^T @ gy
				local W_grad, b_grad, x_grad
				if W._requiresGrad then
					W_grad = ensureGrad(W)
				end
				if b._requiresGrad then
					b_grad = ensureGrad(b)
				end
				if x._requiresGrad then
					x_grad = ensureGrad(x)
				end
				task.desynchronize()
				local gy_store = gy._storage
				local B = gy._shape[2] -- Batch size (N)
				if b_grad then
					local bg_store = b_grad._storage
					for i = 1, outF do
						local acc = 0
						local offset = (i - 1) * B
						for j = 1, B do
							acc += gy_store[offset + j]
						end
						bg_store[i] += acc
					end
				end
				-- dW = gy @ x.T
				-- gy: {outF, B}, x: {inF, B} -> dW: {outF, inF}
				-- dW[i, k] = sum_j ( gy[i, j] * x[k, j] )  <-- implicit transpose of x
				if W_grad then
					local wg_store = W_grad._storage
					local x_store = x._storage
					for i = 1, outF do
						local gy_offset = (i - 1) * B
						local w_offset = (i - 1) * inF
						for k = 1, inF do
							local acc = 0
							local x_offset = (k - 1) * B
							for j = 1, B do
								-- gy[i, j] * x[k, j]
								acc += gy_store[gy_offset + j] * x_store[x_offset + j]
							end
							wg_store[w_offset + k] += acc
						end
					end
				end
				-- dx = W.T @ gy
				-- W: {outF, inF}, gy: {outF, B} -> dx: {inF, B}
				-- dx[k, j] = sum_i ( W[i, k] * gy[i, j] )
				if x_grad then
					local xg_store = x_grad._storage
					local W_store = W._storage
					for k = 1, inF do
						local x_offset = (k - 1) * B
						for j = 1, B do
							local acc = 0
							for i = 1, outF do
								-- W[i, k] => (i-1)*inF + k
								-- gy[i, j] => (i-1)*B + j
								acc += W_store[(i - 1) * inF + k] * gy_store[(i - 1) * B + j]
							end
							xg_store[x_offset + j] += acc
						end
					end
				end
				task.synchronize()
			end)
		end,
		parameters = function(_)
			return { W, b }
		end,
	}
end
