--!native
--!optimize 2
local Tensor = require("../Tensor")
local Tape = require("../autograd/Tape")
local Types = require("../Types")
local Util = require("../Util")
local assert = Util.Assert

local Stabilizers_ok, Stabilizers = pcall(require, "../Utils/Stabilizers")

local function softUpdateMaybe(tp, fp, tau: number?)
	if tau and Stabilizers_ok then
		Stabilizers.softUpdate(tp, fp, tau)
	end
end

local function softmaxAtoms(logits: Types.Tensor, A: number, Z: number)
	local B = logits._shape[2]
	local out = Tensor.zeros({ A * Z, B }, logits._dtype, false)
	task.desynchronize()
	for j = 1, B do
		for a = 1, A do
			local base = ((a - 1) * Z)
			local m = -1 / 0
			for z = 1, Z do
				local v = logits._storage[(base + z - 1) * B + j]
				if v > m then
					m = v
				end
			end
			local es = 0
			for z = 1, Z do
				local e = math.exp(logits._storage[(base + z - 1) * B + j] - m)
				out._storage[(base + z - 1) * B + j] = e
				es += e
			end
			local inv = 1 / es
			for z = 1, Z do
				out._storage[(base + z - 1) * B + j] *= inv
			end
		end
	end
	task.synchronize()
	return out
end

local function qExpect(P: Types.Tensor, A: number, Z: number, zVals: { number })
	local B = P._shape[2]
	local Q = Tensor.zeros({ A, B }, P._dtype, false)
	task.desynchronize()
	for j = 1, B do
		for a = 1, A do
			local base = (a - 1) * Z
			local acc = 0
			for z = 1, Z do
				acc += P._storage[(base + z - 1) * B + j] * zVals[z]
			end
			Q._storage[(a - 1) * B + j] = acc
		end
	end
	task.synchronize()
	return Q
end

local function project(
	zNext: { number },
	pNext: Types.Tensor,
	rewards: { number },
	dones: { number },
	gamma: number,
	A: number,
	Z: number,
	B: number
)
	local vmin, vmax = zNext[1], zNext[#zNext]
	local dz = (vmax - vmin) / (Z - 1)
	local Tproj = Tensor.zeros({ A * Z, B })
	return function(j: number, aNext: number)
		local r, done = rewards[j], dones[j]
		for z = 1, Z do
			local tz = done == 1 and r or (r + gamma * zNext[z])
			local tzClamped = math.max(vmin, math.min(vmax, tz))
			local b = (tzClamped - vmin) / dz
			local l = math.floor(b)
			local u = math.ceil(b)
			for m = l, u do
				if m >= 0 and m <= (Z - 1) then
					local w = (m == u and (b - m)) or (m == l and (1 - (b - m))) or 0
					if w ~= 0 then
						local idx = ((aNext - 1) * Z + (m + 1) - 1) * B + j
						Tproj._storage[idx] += w * pNext._storage[((aNext - 1) * Z + z - 1) * B + j]
					end
				end
			end
		end
	end,
		Tproj
end

return function(cfg)
	local A = assert(cfg.actionDim, "actionDim required")
	local Z = cfg.numAtoms or 51
	local vmin = cfg.vmin or -10
	local vmax = cfg.vmax or 10
	local gamma = cfg.gamma or 0.99
	local dz = (vmax - vmin) / (Z - 1)
	local zVals = table.create(Z)
	for i = 1, Z do
		zVals[i] = vmin + (i - 1) * dz
	end

	local policy = cfg.modelFactory()
	local target = cfg.modelFactory()
	local function copyParams()
		local fp, tp = policy:parameters(), target:parameters()
		task.desynchronize()
		for i = 1, #fp do
			for k = 1, #fp[i]._storage do
				tp[i]._storage[k] = fp[i]._storage[k]
			end
		end
		task.synchronize()
	end
	copyParams()
	local opt = cfg.optimizerFactory(policy:parameters())
	local step = 0

	local function epsAt(n: number)
		local d = cfg.epsilonDecay or 15000
		local e0, e1 = cfg.epsilonStart or 1.0, cfg.epsilonEnd or 0.10
		local k = math.exp(-n / d)
		return e1 + (e0 - e1) * k
	end

	local function pickAction(state)
		-- logits {A*Z,1} -> probs -> expected Q per action -> argmax
		local logits = policy:forward(state)
		local P = softmaxAtoms(logits, A, Z)
		local Q = qExpect(P, A, Z, zVals)
		-- argmax
		local best, ai = -1 / 0, 1
		for a = 1, A do
			local v = Q._storage[(a - 1) * 1 + 1]
			if v > best then
				best = v
				ai = a
			end
		end
		return ai
	end

	return {
		optimizer = opt,
		
		getPolicy = function(_)
			return policy
		end,

		parameters = function(_)
			return policy:parameters()
		end,

		act = function(_, state, n)
			local eps = epsAt(n)
			if math.random() < eps then
				return math.random(1, A)
			end
			return pickAction(state)
		end,

		observe = function(_, t)
			if cfg.replay then
				cfg.replay:push(t)
			end
		end,

		trainStep = function(_)
			if not cfg.replay or not cfg.replay:canSample(cfg.batchSize) then
				return nil
			end
			opt:zeroGrad()
			local batch = cfg.replay:sample(cfg.batchSize)
			local S, Aidx, R, NS, D = batch.S, batch.A, batch.R, batch.NS, batch.D
			local B = cfg.batchSize

			-- logits {A*Z,B}
			local logits = policy:forward(S)
			local P = softmaxAtoms(logits, A, Z)

			local logitsNext_policy = policy:forward(NS)
			local Pnext_policy = softmaxAtoms(logitsNext_policy, A, Z)
			local Qnext_policy = qExpect(Pnext_policy, A, Z, zVals)
			local aNext = table.create(B, 1)
			task.desynchronize()
			for j = 1, B do
				local best, ai = -1 / 0, 1
				for a = 1, A do
					local v = Qnext_policy._storage[(a - 1) * B + j]
					if v > best then
						best = v
						ai = a
					end
				end
				aNext[j] = ai
			end
			task.synchronize()

			local logitsNext_target = target:forward(NS)
			local Pnext_target = softmaxAtoms(logitsNext_target, A, Z)

			local buildOne, Tproj = project(zVals, Pnext_target, R, D, gamma, A, Z, B)
			for j = 1, B do
				buildOne(j, aNext[j])
			end

			local Psel = Tensor.zeros({ Z, B })
			task.desynchronize()
			for j = 1, B do
				local a = Aidx[j]
				for z = 1, Z do
					Psel._storage[(z - 1) * B + j] = P._storage[((a - 1) * Z + z - 1) * B + j]
				end
			end
			task.synchronize()

			local Tsel = Tensor.zeros({ Z, B })
			task.desynchronize()
			for j = 1, B do
				local a = Aidx[j]
				for z = 1, Z do
					Tsel._storage[(z - 1) * B + j] = Tproj._storage[((a - 1) * Z + z - 1) * B + j]
				end
			end
			task.synchronize()

			local grad = Tensor.zeros({ A * Z, B })
			local loss = 0
			task.desynchronize()
			for j = 1, B do
				for z = 1, Z do
					local p = math.max(Psel._storage[(z - 1) * B + j], 1e-12)
					local t = Tsel._storage[(z - 1) * B + j]
					loss += -t * math.log(p)
				end
			end
			for j = 1, B do
				local a = Aidx[j]
				for z = 1, Z do
					local p = Psel._storage[(z - 1) * B + j]
					local t = Tsel._storage[(z - 1) * B + j]
					grad._storage[((a - 1) * Z + z - 1) * B + j] = (p - t) / B
				end
			end
			task.synchronize()

			Tape.backwardFrom(logits, grad)
			if Stabilizers_ok then
				Stabilizers.clipGradGlobalNorm(policy:parameters(), 3.0)
			end
			opt:step()

			step += 1
			if cfg.tau then
				softUpdateMaybe(target:parameters(), policy:parameters(), cfg.tau)
			elseif cfg.targetSyncInterval and (step % cfg.targetSyncInterval == 0) then
				copyParams()
			end
			return loss / B
		end,

		loadParameters = function(_, snapshot, strict)
			local fp = policy:parameters()
			if strict ~= false then
				assert(#fp == #snapshot, "snapshot size mismatch")
				for i = 1, #fp do
					assert(#fp[i]._storage == #snapshot[i], "tensor size mismatch at " .. i)
				end
			end
			local n = math.min(#fp, #snapshot)
			task.desynchronize()
			for i = 1, n do
				local p, s = fp[i], snapshot[i]
				local m = math.min(#p._storage, #s)
				for k = 1, m do
					p._storage[k] = s[k]
				end
			end
			task.synchronize()
			local fp2, tp = policy:parameters(), target:parameters()
			task.desynchronize()
			for i = 1, #fp2 do
				for k = 1, #fp2[i]._storage do
					tp[i]._storage[k] = fp2[i]._storage[k]
				end
			end
			task.synchronize()
		end,
	}
end
