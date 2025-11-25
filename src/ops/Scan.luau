--!native
--!optimize 2
local Tensor = require("../Tensor")
local Types = require("../Types")
local Tape = require("../autograd/Tape")

local Scan = {}

local function ensureGrad(t: Types.Tensor): Types.Tensor?
	if not t._grad then
		t._grad = Tensor.zeros(t._shape, t._dtype, false)
	end
	return t._grad
end

function Scan.selectiveScan(
	u: Types.Tensor,
	delta: Types.Tensor,
	A: Types.Tensor,
	B: Types.Tensor,
	C: Types.Tensor,
	D: Types.Tensor?
): Types.Tensor
	local batch, seqLen, dim = u._shape[1], u._shape[2], u._shape[3]
	local stateDim = A._shape[2] -- N
	local requiresGrad = u._requiresGrad
		or delta._requiresGrad
		or A._requiresGrad
		or B._requiresGrad
		or C._requiresGrad
		or (D and D._requiresGrad)
	local out = Tensor.zeros({ batch, seqLen, dim }, u._dtype, requiresGrad)
	local us = u._storage
	local ds = delta._storage
	local As = A._storage
	local Bs = B._storage
	local Cs = C._storage
	local outs = out._storage
	local Ds = D and D._storage
	local x_storage
	if requiresGrad then
		x_storage = table.create(batch * seqLen * dim * stateDim, 0)
	end
	task.desynchronize()
	for b = 1, batch do
		local x_state = table.create(dim * stateDim, 0)
		local b_offset = (b - 1) * seqLen * dim

		for t = 1, seqLen do
			local t_offset = (t - 1) * dim
			-- B/C: (b-1)*L*N + (t-1)*N
			local bc_base = (b - 1) * seqLen * stateDim + (t - 1) * stateDim
			for d = 1, dim do
				local idx = b_offset + t_offset + d
				local u_val = us[idx]
				local delta_val = ds[idx]
				local y_val = 0
				for n = 1, stateDim do
					local state_idx = (d - 1) * stateDim + n
					local a_val = As[state_idx] -- A is {D, N}
					local b_val = Bs[bc_base + n] -- B is {B, L, N}
					local dA = math.exp(delta_val * a_val)
					local dB = delta_val * b_val
					local prev_x = x_state[state_idx]
					local new_x = dA * prev_x + dB * u_val
					x_state[state_idx] = new_x
					if x_storage then
						local store_idx = ((b - 1) * seqLen + (t - 1)) * dim * stateDim + (d - 1) * stateDim + n
						x_storage[store_idx] = new_x
					end
					local c_val = Cs[bc_base + n]
					y_val += c_val * new_x
				end
				if Ds then
					y_val += Ds[d] * u_val
				end
				outs[idx] = y_val
			end
		end
	end
	task.synchronize()
	if not requiresGrad then
		return out
	end
	local parents = { u, delta, A, B, C }
	if D then
		table.insert(parents, D)
	end
	return Tape.makeNode(out, parents, function(gy)
		local gu = u._requiresGrad and ensureGrad(u)
		local gdelta = delta._requiresGrad and ensureGrad(delta)
		local gA = A._requiresGrad and ensureGrad(A)
		local gB = B._requiresGrad and ensureGrad(B)
		local gC = C._requiresGrad and ensureGrad(C)
		local gD = (D and D._requiresGrad) and ensureGrad(D)
		local gus = gu and gu._storage
		local gds = gdelta and gdelta._storage
		local gAs = gA and gA._storage
		local gBs = gB and gB._storage
		local gCs = gC and gC._storage
		local gDs = gD and gD._storage
		local gys = gy._storage
		task.desynchronize()
		for b = 1, batch do
			local dx_next = table.create(dim * stateDim, 0)
			for t = seqLen, 1, -1 do
				local b_offset = (b - 1) * seqLen * dim
				local t_offset = (t - 1) * dim
				local bc_base = (b - 1) * seqLen * stateDim + (t - 1) * stateDim
				for d = 1, dim do
					local idx = b_offset + t_offset + d
					local gy_val = gys[idx]
					local u_val = us[idx]
					local delta_val = ds[idx]
					local d_delta_acc = 0
					local d_u_acc = 0
					if gDs then
						gDs[d] += gy_val * u_val
						d_u_acc += gy_val * Ds[d]
					elseif Ds then
						d_u_acc += gy_val * Ds[d]
					end
					for n = 1, stateDim do
						local state_idx = (d - 1) * stateDim + n
						local a_val = As[state_idx]
						local b_val = Bs[bc_base + n]
						local c_val = Cs[bc_base + n]
						local store_idx = ((b - 1) * seqLen + (t - 1)) * dim * stateDim + (d - 1) * stateDim + n
						local x_curr = x_storage[store_idx]
						local x_prev = 0
						if t > 1 then
							local prev_store_idx = ((b - 1) * seqLen + (t - 2)) * dim * stateDim
								+ (d - 1) * stateDim
								+ n
							x_prev = x_storage[prev_store_idx]
						end
						local dA = math.exp(delta_val * a_val)
						local dB = delta_val * b_val
						local d_x_curr = gy_val * c_val + dx_next[state_idx]
						if gCs then
							gCs[bc_base + n] += gy_val * x_curr
						end
						dx_next[state_idx] = d_x_curr * dA
						local d_dA = d_x_curr * x_prev
						local d_argA = d_dA * dA -- deriv of exp is exp
						if gAs then
							gAs[state_idx] += d_argA * delta_val
						end
						d_delta_acc += d_argA * a_val
						local d_dB = d_x_curr * u_val
						if gBs then
							gBs[bc_base + n] += d_dB * delta_val
						end
						d_delta_acc += d_dB * b_val
						-- u: dL/du += dL/dx_curr * dB
						d_u_acc += d_x_curr * dB
					end
					if gds then
						gds[idx] = d_delta_acc
					end
					if gus then
						gus[idx] = d_u_acc
					end
				end
			end
		end
		task.synchronize()
	end)
end

return Scan
