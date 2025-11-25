--!native
--!optimize 2
local Tensor = require("../../Tensor")
local Tape = require("../../autograd/Tape")
local Types = require("../../Types")
local Scan = require("../../ops/Scan")
local Initializer = require("../../Initializer")
local BLAS = require("../../ops/BLAS")
local Math = require("../../ops/Math")
local Activations = require("../../nn/Activations")

local function ensureGrad(t: Types.Tensor): Types.Tensor?
	if not t._grad then
		t._grad = Tensor.zeros(t._shape, t._dtype, false)
	end
	return t._grad
end

local function depthwiseConv1d(x: Types.Tensor, weight: Types.Tensor): Types.Tensor
	-- x: {B, L, D}
	if not Tensor.is_contiguous(x) then
		x = Tensor.contiguous(x)
	end
	if not Tensor.is_contiguous(weight) then
		weight = Tensor.contiguous(weight)
	end
	local B, L, D = x._shape[1], x._shape[2], x._shape[3]
	local K = weight._shape[3]
	local y = Tensor.zeros({ B, L, D }, x._dtype, x._requiresGrad or weight._requiresGrad)
	task.desynchronize()
	local xs = x._storage
	local ws = weight._storage
	local ys = y._storage
	for b = 0, B - 1 do
		local b_offset = b * L * D
		for l = 0, L - 1 do
			local l_offset = b_offset + l * D
			for d = 0, D - 1 do
				local sum = 0
				for k = 0, K - 1 do
					if l - k >= 0 then
						local x_idx = b_offset + (l - k) * D + d + 1
						local w_idx = d * K + k + 1
						sum += xs[x_idx] * ws[w_idx]
					end
				end
				ys[l_offset + d + 1] = sum
			end
		end
	end
	task.synchronize()
	return Tape.makeNode(y, { x, weight }, function(gy)
		if not x._requiresGrad and not weight._requiresGrad then
			return
		end
		local gx = x._requiresGrad and ensureGrad(x)
		local gw = weight._requiresGrad and ensureGrad(weight)
		local gys = gy._storage
		task.desynchronize()
		local gxs = gx and gx._storage
		local gws = gw and gw._storage
		for b = 0, B - 1 do
			local b_offset = b * L * D
			for l = 0, L - 1 do
				local l_offset = b_offset + l * D
				for d = 0, D - 1 do
					local gy_val = gys[l_offset + d + 1]
					if gy_val == 0 then
						continue
					end
					for k = 0, K - 1 do
						if l - k >= 0 then
							local x_idx = b_offset + (l - k) * D + d + 1
							local w_idx = d * K + k + 1
							if gxs then
								gxs[x_idx] += gy_val * ws[w_idx]
							end
							if gws then
								gws[w_idx] += gy_val * xs[x_idx]
							end
						end
					end
				end
			end
		end
		task.synchronize()
	end)
end

local function linear(x: Types.Tensor, weight: Types.Tensor, bias: Types.Tensor?): Types.Tensor
	local inFeat = weight._shape[2]
	local outFeat = weight._shape[1]
	local xShape, xShape_len = x._shape, #x._shape
	local lastDim = xShape[xShape_len]
	if lastDim ~= inFeat then
		error(`Linear: input dimension mismatch. Expected {inFeat}, got {lastDim}`)
	end
	local flattenedDim = 1
	for i = 1, xShape_len - 1 do
		flattenedDim *= xShape[i]
	end
	local xFlat = Tensor.reshape(x, { flattenedDim, inFeat })
	local wT = Tensor.transpose(weight, 1, 2)
	local outFlat = BLAS.matmul(xFlat, wT)
	if bias then
		local outWithBias = Tensor.empty(outFlat._shape, outFlat._dtype, outFlat._requiresGrad or bias._requiresGrad)
		local b_storage = bias._storage
		local out_storage = outFlat._storage
		local res_storage = outWithBias._storage
		local outSize = #out_storage
		task.desynchronize()
		for i = 0, outSize - 1 do
			local c = (i % outFeat) + 1
			res_storage[i + 1] = out_storage[i + 1] + b_storage[c]
		end
		task.synchronize()
		if outWithBias._requiresGrad then
			outFlat = Tape.makeNode(outWithBias, { outFlat, bias }, function(gy)
				if outFlat._requiresGrad then
					local gout = ensureGrad(outFlat)
					local gys = gy._storage
					local gos = gout._storage
					task.desynchronize()
					for i = 1, #gys do
						gos[i] += gys[i]
					end
					task.synchronize()
				end
				if bias._requiresGrad then
					local gb = ensureGrad(bias)
					local gys = gy._storage
					local gbs = gb._storage
					task.desynchronize()
					for i = 0, #gys - 1 do
						local c = (i % outFeat) + 1
						gbs[c] += gys[i + 1]
					end
					task.synchronize()
				end
			end)
		else
			outFlat = outWithBias
		end
	end
	local outShape = table.clone(xShape)
	outShape[#outShape] = outFeat
	return Tensor.reshape(outFlat, outShape)
end

local function MambaBlock(dModel: number, dState: number?, dConv: number?, expand: number?)
	dState = dState or 16
	dConv = dConv or 4
	expand = expand or 2
	local dInner = dModel * (expand :: number)
	local dtRank = math.ceil(dModel / 16)
	local inProj = Tensor.empty({ dInner * 2, dModel }, "f64", true)
	Initializer.xavierUniform(inProj)
	local convWeight = Tensor.empty({ dInner, 1, dConv :: number }, "f64", true)
	Initializer.kaimingUniform(convWeight)
	local A_log = Tensor.empty({ dInner, dState :: number }, "f64", true)
	Initializer.kaimingUniform(A_log)
	local D = Tensor.ones({ dInner }, "f64", true)
	local xProj = Tensor.empty({ dtRank + (dState :: number) * 2, dInner }, "f64", true)
	local dtProj = Tensor.empty({ dInner, dtRank }, "f64", true)
	Initializer.xavierUniform(xProj)
	Initializer.xavierUniform(dtProj)
	local outProj = Tensor.empty({ dModel, dInner }, "f64", true)
	Initializer.xavierUniform(outProj)
	return {
		forward = function(_, u: Types.Tensor): Types.Tensor
			local _B_size, _L_size, D_size = u._shape[1], u._shape[2], u._shape[3]
			local xz = linear(u, inProj)
			local x = Tensor.slice(xz, 3, 1, dInner)
			local z = Tensor.slice(xz, 3, dInner + 1, dInner * 2)
			local x_conv = depthwiseConv1d(x, convWeight)
			local x_act = Activations.SiLU(x_conv)
			local x_dbl = linear(x_act, xProj)
			local dt_rank = Tensor.slice(x_dbl, 3, 1, dtRank)
			local B_ssm = Tensor.slice(x_dbl, 3, dtRank + 1, dtRank + (dState :: number))
			local C_ssm = Tensor.slice(x_dbl, 3, dtRank + (dState :: number) + 1, dtRank + (dState :: number) * 2)
			local dt = linear(dt_rank, dtProj)
			dt = Activations.Softplus(dt)
			local A = Math.scalarMul(Math.exp(A_log), -1)
			local y = Scan.selectiveScan(x_act, dt, A, B_ssm, C_ssm, D)
			local z_act = Activations.SiLU(z)
			local y_gated = Math.mul(y, z_act)
			local out = linear(y_gated, outProj)
			if D_size == dModel then
				return Math.add(out, u)
			end
			return out
		end,
		parameters = function()
			return { inProj, convWeight, A_log, D, xProj, dtProj, outProj }
		end,
	}
end

return MambaBlock
