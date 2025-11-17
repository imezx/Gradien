--!optimize 2
local Tensor = require("../Tensor")
local Tape = require("../autograd/Tape")
local Losses = require("../nn/Losses")
local Util = require("../Util")
local assert = Util.Assert

-- argmax over actions per column (batch)
local function maxPerColumn(Q)
	task.desynchronize()
	local A, B = Q._shape[1], Q._shape[2]
	local maxv = table.create(B, -1/0)
	local arg = table.create(B, 1)
	for j = 1, B do
		local best, ai = -1/0, 1
		for i = 1, A do
			local v = Q._storage[(i - 1) * B + j]
			if v > best then best, ai = v, i end
		end
		maxv[j], arg[j] = best, ai
	end
	task.synchronize()
	return maxv, arg
end

-- Pack a list of transitions into a batch {S,A,R,NS,D}
local function packBatch(transitions)
	task.desynchronize()
	local n = #transitions
	local sd = transitions[1].state._shape[1]
	local S = Tensor.zeros({ sd, n })
	local NS = Tensor.zeros({ sd, n })
	local A = table.create(n, 1)
	local R = table.create(n, 0.0)
	local D = table.create(n, false)
	for j = 1, n do
		local t = transitions[j]
		A[j], R[j], D[j] = t.action, t.reward, t.done
		for k = 1, sd do
			S._storage[(k - 1) * n + j] = t.state._storage[k]
			NS._storage[(k - 1) * n + j] = t.nextState._storage[k]
		end
	end
	task.synchronize()
	return S, A, R, NS, D
end

local function asArray(entry)
	if type(entry) == "table" then
		if entry.data then return entry.data end
		if entry[1] ~= nil then return entry end
	end
	error("Bad snapshot entry; expected {data=...} or numeric array")
end

return function(cfg)
	local batchSize = cfg.batchSize or 1
	local gamma = cfg.gamma or 0.99
	local epsStart = cfg.epsilonStart or 1.0
	local epsEnd = cfg.epsilonEnd or 0.05
	local epsDecay = cfg.epsilonDecay or 1000
	local targetSyncEvery = cfg.targetSyncInterval or 200
	local tau = cfg.tau
	local actionDim = assert(cfg.actionDim, "cfg.actionDim required")

	local policy = cfg.modelFactory()
	local target = cfg.modelFactory()

	local function copyParams()
		task.desynchronize()
		local fps = policy:parameters()
		local tps = target:parameters()
		for i = 1, #fps do
			for k = 1, #fps[i]._storage do
				tps[i]._storage[k] = fps[i]._storage[k]
			end
		end
		task.synchronize()
	end
	copyParams()
	
	local function softUpdate(alpha: number)
		if not alpha or alpha <= 0 then return end
		local fp, tp = policy:parameters(), target:parameters()
		task.desynchronize()
		for i = 1, #fp do
			for k = 1, #fp[i]._storage do
				local tv = tp[i]._storage[k]
				local pv = fp[i]._storage[k]
				tp[i]._storage[k] = tv + alpha * (pv - tv)
			end
		end
		task.synchronize()
	end

	local opt = cfg.optimizerFactory(policy:parameters())
	local step = 0

	local onlineBuf = {}
	local onlineMax = cfg.onlineMaxBatch or math.max(1, batchSize)

	local function epsAt(n)
		local k = math.exp(-n / epsDecay)
		return epsEnd + (epsStart - epsEnd) * k
	end

	local function tdUpdate(S, A, R, NS, D)
		task.desynchronize()
		local B = #A
		local Q = policy:forward(S) -- {A,B}
		local Qn = target:forward(NS) -- {A,B}
		local maxn = (maxPerColumn(Qn))

		local Y = Tensor.zeros({ actionDim, B })
		for i = 1, #Q._storage do
			Y._storage[i] = Q._storage[i]
		end
		for j = 1, B do
			local target_j = R[j] + gamma * (D[j] and 0 or maxn[j])
			Y._storage[(A[j] - 1) * B + j] = target_j
		end

		local loss, grad = Losses.mse_backward(Q, Y)
		Tape.backwardFrom(Q, grad)
		opt:step()

		step += 1
		if tau and tau > 0 then
			softUpdate(tau)
		elseif targetSyncEvery > 0 and step % targetSyncEvery == 0 then
			copyParams()
		end
		task.synchronize()
		return loss
	end

	return {
		getPolicy = function(_)
			return policy
		end,

		parameters = function(_)
			return policy:parameters()
		end,
		
		act = function(_, state, n)
			local eps = epsAt(n or step)
			if math.random() < eps then
				return math.random(1, actionDim)
			end
			task.desynchronize()
			local q = policy:forward(state)
			local _, arg = maxPerColumn(q)
			task.synchronize()
			return arg[1]
		end,

		observe = function(_, t)
			if cfg.replay then
				cfg.replay:push(t)
				return
			end
			table.insert(onlineBuf, t)
			if #onlineBuf > onlineMax then
				table.remove(onlineBuf, 1)
			end
		end,

		trainStep = function(_)
			if cfg.replay then
				if not cfg.replay:canSample(batchSize) then
					return nil
				end
				local B = cfg.replay:sample(batchSize)
				return tdUpdate(B.S, B.A, B.R, B.NS, B.D)
			end
			local lenBuf = #onlineBuf
			if lenBuf == 0 then
				return nil
			end
			task.desynchronize()
			local n = math.min(batchSize, lenBuf)
			local tmp = table.create(n)
			for i = lenBuf - n + 1, lenBuf do
				table.insert(tmp, onlineBuf[i])
			end
			local S, A, R, NS, D = packBatch(tmp)
			if not cfg.onlineKeep then
				table.clear(onlineBuf)
			end
			task.synchronize()
			return tdUpdate(S, A, R, NS, D)
		end,
		
		loadParameters = function(_, snapshot, strict: boolean?)
			local fp = policy:parameters()
			local n = #fp
			assert(n > 0, "No parameters in policy")

			local arr = table.create(n)
			for i = 1, n do arr[i] = asArray(snapshot[i]) end

			if strict ~= false then
				assert(#arr == n, "snapshot size mismatch")
				for i = 1, n do
					assert(#fp[i]._storage == #arr[i],
						("tensor size mismatch at %d, got %d:%d")
							:format(i, #fp[i]._storage, #arr[i]))
				end
			end

			task.desynchronize()
			for i = 1, n do
				local p, s = fp[i], arr[i]
				local m = math.min(#p._storage, #s)
				for k = 1, m do p._storage[k] = s[k] end
			end
			task.synchronize()

			-- refresh target (hard sync)
			local tp = target:parameters()
			task.desynchronize()
			for i = 1, n do
				local ps, ts = fp[i]._storage, tp[i]._storage
				for k = 1, #ps do ts[k] = ps[k] end
			end
			task.synchronize()
		end,
	}
end
