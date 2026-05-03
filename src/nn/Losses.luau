--!native
--!optimize 2
local Tensor = require("../Tensor")
local Softmax = require("./Softmax")
local Types = require("../Types")
local Util = require("../Util")
local assert = Util.Assert

local L = {}

--@param logits Tensor {C,B}
--@param targetIdx {number} -- {B}
--@param alpha number? -- default 1.0
--@param gamma number? -- default 2.0
--@return number, Tensor -- mean loss, grad dL/dlogits
function L.focal_backward(logits: Types.Tensor, targetIdx: { number }, alpha: number?, gamma: number?)
	alpha = alpha or 1.0
	gamma = gamma or 2.0
	local C, B = logits._shape[1], logits._shape[2]
	assert(#targetIdx == B, "Focal: target length must equal batch")

	task.desynchronize()
	-- softmax
	local sm = Tensor.zeros({ C, B }, false)
	for j = 1, B do
		local m = -1 / 0
		for i = 1, C do
			local z = logits._storage[(i - 1) * B + j]
			if z > m then
				m = z
			end
		end
		local esum = 0
		for i = 1, C do
			local e = math.exp(logits._storage[(i - 1) * B + j] - m)
			sm._storage[(i - 1) * B + j] = e
			esum += e
		end
		for i = 1, C do
			sm._storage[(i - 1) * B + j] /= esum
		end
	end

	-- loss + grad
	local lossSum = 0
	local grad = Tensor.zeros({ C, B }, false)

	for j = 1, B do
		local y = targetIdx[j]
		local pt = sm._storage[(y - 1) * B + j]
		local oneMinusPt = 1 - pt
		local w = alpha * (oneMinusPt ^ gamma)
		-- loss_j = - w * log(pt)
		lossSum += -w * math.log(math.max(pt, 1e-12))

		for i = 1, C do
			local p = sm._storage[(i - 1) * B + j]
			local y_i = (i == y) and 1 or 0
			-- d/dz focal = w * (p - y) + extra term on true class:
			local g = w * (p - y_i)
			if i == y then
				-- extra term: -(alpha * gamma * (1-pt)^(gamma-1) * (-dp/dz) * log(pt))
				-- for dp/dz on true class ≈ pt*(1-pt)
				g += -alpha * gamma * (oneMinusPt ^ math.max(gamma - 1, 0)) * (pt * (1 - pt)) * math.log(
					math.max(pt, 1e-12)
				)
			end
			grad._storage[(i - 1) * B + j] = g / B
		end
	end
	task.synchronize()
	return lossSum / B, grad
end

function L.mse_backward(pred: Types.Tensor, target: Types.Tensor)
	task.desynchronize()
	local B = pred._shape[#pred._shape]
	local grad = Tensor.zeros(pred._shape)
	local loss = 0
	for i = 1, #pred._storage do
		local d = pred._storage[i] - target._storage[i]
		loss += d * d
		grad._storage[i] = (2 / B) * d
	end
	task.synchronize()
	return loss / B, grad
end

function L.huber_backward(pred: Types.Tensor, target: Types.Tensor, delta: number?)
	task.desynchronize()
	delta = delta or 1.0
	local B = pred._shape[#pred._shape]
	local grad = Tensor.zeros(pred._shape)
	local loss = 0
	for i = 1, #pred._storage do
		local d = pred._storage[i] - target._storage[i]
		local ad = math.abs(d)
		if ad <= delta then
			loss += 0.5 * d * d
			grad._storage[i] = d / B
		else
			loss += delta * (ad - 0.5 * delta)
			grad._storage[i] = (delta * (d >= 0 and 1 or -1)) / B
		end
	end
	task.synchronize()
	return loss / B, grad
end

function L.softmaxCE_backward(logits: Types.Tensor, targets: { number }, smoothing: number?)
	task.desynchronize()
	local C, B = logits._shape[1], logits._shape[2]
	local eps = math.clamp(smoothing or 0, 0, 1) -- label smoothing in [0,1]
	local P = Softmax.forward(logits) -- {C,B} probs
	local grad = Tensor.zeros({ C, B })
	local loss = 0

	-- "uniform" smoothing: y' = (1-ε) * one_hot + ε/C
	local off = eps / C

	for j = 1, B do
		local t = math.clamp(targets[j], 1, C)
		local on = 1 - eps + off
		for i = 1, C do
			local idx = (i - 1) * B + j
			local pij = math.max(P._storage[idx], 1e-12)
			local yij = (i == t) and on or off
			loss += -yij * math.log(pij)
			grad._storage[idx] = P._storage[idx] - yij
		end
	end
	for k = 1, #grad._storage do
		grad._storage[k] /= B
	end
	task.synchronize()
	return loss / B, grad
end

function L.bceWithLogits_backward(logits: Types.Tensor, targets: { number })
	task.desynchronize()
	local B = logits._shape[2]
	local grad = Tensor.zeros(logits._shape)
	local loss = 0
	for j = 1, B do
		local z = logits._storage[j]
		local y = math.clamp(targets[j], 0, 1)
		local p = 1 / (1 + math.exp(-z))
		loss += -(y * math.log(math.max(p, 1e-12)) + (1 - y) * math.log(math.max(1 - p, 1e-12)))
		grad._storage[j] = (p - y) / B
	end
	task.synchronize()
	return loss / B, grad
end

function L.l1_backward(pred: Types.Tensor, target: Types.Tensor)
	task.desynchronize()
	local grad = Tensor.zeros(pred._shape)
	local loss = 0
	for i = 1, #pred._storage do
		local d = pred._storage[i] - target._storage[i]
		loss += math.abs(d)
		grad._storage[i] = (d >= 0) and 1 or -1
	end
	for i = 1, #grad._storage do
		grad._storage[i] /= pred._shape[2]
	end
	task.synchronize()
	return loss / pred._shape[2], grad
end

function L.kl_div_backward(pred: Types.Tensor, target: Types.Tensor, fromLogits: boolean?)
	local C, B = pred._shape[1], pred._shape[2]
	local logp
	if fromLogits then
		logp = Tensor.zeros({ C, B }, pred._requiresGrad)
		task.desynchronize()
		for j = 1, B do
			local m = -1 / 0
			for i = 1, C do
				local z = pred._storage[(i - 1) * B + j]
				if z > m then
					m = z
				end
			end
			local esum = 0
			for i = 1, C do
				esum += math.exp(pred._storage[(i - 1) * B + j] - m)
			end
			local logZ = m + math.log(esum)
			for i = 1, C do
				local idx = (i - 1) * B + j
				logp._storage[idx] = pred._storage[idx] - logZ
			end
		end
		task.synchronize()
	else
		logp = pred
	end
	-- loss = mean_j sum_i T_ij * (logT_ij - logP_ij)
	local lossSum = 0.0
	local grad = Tensor.zeros({ C, B }, false)
	task.desynchronize()
	for j = 1, B do
		for i = 1, C do
			local idx = (i - 1) * B + j
			local t = math.max(target._storage[idx], 1e-12)
			lossSum += t * (math.log(t) - logp._storage[idx])
		end
	end
	for j = 1, B do
		local denom = 0
		for i = 1, C do
			denom += math.exp(logp._storage[(i - 1) * B + j])
		end
		for i = 1, C do
			local idx = (i - 1) * B + j
			local p = math.exp(logp._storage[idx]) / denom
			grad._storage[idx] = (p - target._storage[idx]) / B
		end
	end
	task.synchronize()
	return lossSum / B, grad
end

L.cross_entropy_backward = L.softmaxCE_backward

return L
