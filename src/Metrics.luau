--!native
--!optimize 2
local Types = require("./Types")
local M = {}

function M.accuracy(logits: Types.Tensor, targets: Types.Tensor): number
	task.desynchronize()
	local C, B = logits._shape[1], logits._shape[2]
	local correct = 0
	for j = 1, B do
		local best = -1 / 0
		local arg = 1
		for i = 1, C do
			local v = logits._storage[(i - 1) * B + j]
			if v > best then
				best = v
				arg = i
			end
		end
		if arg == targets[j] then
			correct += 1
		end
	end
	task.synchronize()
	return correct / B
end

function M.topk(logits: Types.Tensor, targets: Types.Tensor, k: number): number
	task.desynchronize()
	local C, B = logits._shape[1], logits._shape[2]
	local hit = 0
	for j = 1, B do
		local arr = {}
		for i = 1, C do
			arr[i] = logits._storage[(i - 1) * B + j]
		end
		table.sort(arr, function(a, b)
			return a > b
		end)
		local thresh = arr[math.min(k, C)]
		local t = targets[j]
		local ok = false
		for i = 1, C do
			if logits._storage[(i - 1) * B + j] >= thresh and i == t then
				ok = true
			end
		end
		if ok then
			hit += 1
		end
	end
	task.synchronize()
	return hit / B
end

function M.mse(pred: Types.Tensor, target: Types.Tensor): number
	task.desynchronize()
	local s = 0
	for i = 1, #pred._storage do
		local d = pred._storage[i] - target._storage[i]
		s += d * d
	end
	task.synchronize()
	return s / pred._shape[#pred._shape]
end

function M.prf1(pred: Types.Tensor, target: Types.Tensor, C: number): (number, number, number)
	task.desynchronize()
	local yhat: { number }, B: number
	if type(pred) == "table" and pred._storage ~= nil then
		local Cx, Bb = pred._shape[1], pred._shape[2]
		B = Bb
		yhat = table.create(B, 1)
		for j = 1, B do
			local best, arg = -1 / 0, 1
			for i = 1, Cx do
				local v = pred._storage[(i - 1) * B + j]
				if v > best then
					best, arg = v, i
				end
			end
			yhat[j] = arg
		end
	else
		yhat = pred
		B = #yhat
	end
	local tpC: { number }, fpC: { number }, fnC: { number } = {}, {}, {}
	for c = 1, C do
		tpC[c] = 0
		fpC[c] = 0
		fnC[c] = 0
	end
	for j = 1, B do
		local p, t = yhat[j], target[j]
		if p and t and p >= 1 and p <= C and t >= 1 and t <= C then
			if p == t then
				tpC[p] += 1
			else
				fpC[p] += 1
				fnC[t] += 1
			end
		end
	end
	local sumP, sumR, sumF = 0, 0, 0
	for c = 1, C do
		local pc = tpC[c] / (tpC[c] + fpC[c] + 1e-9)
		local rc = tpC[c] / (tpC[c] + fnC[c] + 1e-9)
		local fc = 2 * pc * rc / (pc + rc + 1e-9)
		sumP += pc
		sumR += rc
		sumF += fc
	end
	task.synchronize()
	return sumP / C, sumR / C, sumF / C
end

function M.confusion(pred: Types.Tensor, target: Types.Tensor, C: number): { { number } }
	task.desynchronize()
	local yhat: { number }, B: number
	if type(pred) == "table" and pred._storage ~= nil then
		local Cx, Bb = pred._shape[1], pred._shape[2]
		B = Bb
		yhat = table.create(B, 1)
		for j = 1, B do
			local best, arg = -1 / 0, 1
			for i = 1, Cx do
				local v = pred._storage[(i - 1) * B + j]
				if v > best then
					best, arg = v, i
				end
			end
			yhat[j] = arg
		end
	else
		yhat = pred
		B = #yhat
	end
	local mat: { { number } } = {}
	for i = 1, C do
		mat[i] = table.create(C, 0)
		for j = 1, C do
			mat[i][j] = 0
		end
	end
	for j = 1, B do
		local t, p = target[j], yhat[j]
		if t >= 1 and t <= C and p >= 1 and p <= C then
			mat[t][p] += 1
		end
	end
	task.synchronize()
	return mat
end

return M
