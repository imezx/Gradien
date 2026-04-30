--!native
--!optimize 2
local Tensor = require("../Tensor")
local Tape = require("../autograd/Tape")
local Buffer = require("../Util/Buffer")
local Types = require("../Types")
local Losses = require("../nn/Losses")

-- SAC — Soft Actor-Critic (Haarnoja et al. 2018 / 2019)
-- Off-policy, maximum-entropy RL for continuous actions.
return function(cfg)
	assert(cfg.stateDim, "SAC: cfg.stateDim required")
	assert(cfg.actionDim, "SAC: cfg.actionDim required")
	assert(cfg.actorFactory, "SAC: cfg.actorFactory required")
	assert(cfg.criticFactory, "SAC: cfg.criticFactory required")

	local sDim = cfg.stateDim
	local aDim = cfg.actionDim
	local aHigh = cfg.actionHigh or 1.0
	local gamma = cfg.gamma or 0.99
	local tau = cfg.tau or 0.005
	local batchSize = cfg.batchSize or 64
	local warmup = cfg.warmupSteps or 1000
	local capacity = cfg.capacity or 100000
	local logStdMin = cfg.logStdMin or -20
	local logStdMax = cfg.logStdMax or 2
	local autoAlpha = cfg.autoAlpha == true
	local targetEntropy = cfg.targetEntropy or -aDim
	local alpha = cfg.alpha or 0.2

	local actor = cfg.actorFactory()
	local critic1 = cfg.criticFactory()
	local critic2 = cfg.criticFactory()
	local critic1Target = cfg.criticFactory()
	local critic2Target = cfg.criticFactory()

	-- optimizers
	local actorOpt, criticOpt
	if cfg.actorOptimizerFactory then
		assert(cfg.criticOptimizerFactory, "SAC: criticOptimizerFactory required")
		actorOpt = cfg.actorOptimizerFactory(actor:parameters())
		local cp = {}
		for _, p in ipairs(critic1:parameters()) do
			table.insert(cp, p)
		end
		for _, p in ipairs(critic2:parameters()) do
			table.insert(cp, p)
		end
		criticOpt = cfg.criticOptimizerFactory(cp)
	else
		assert(cfg.optimizerFactory, "SAC: provide optimizerFactory or actor/criticOptimizerFactory")
		actorOpt = cfg.optimizerFactory(actor:parameters())
		local cp = {}
		for _, p in ipairs(critic1:parameters()) do
			table.insert(cp, p)
		end
		for _, p in ipairs(critic2:parameters()) do
			table.insert(cp, p)
		end
		criticOpt = cfg.optimizerFactory(cp)
	end

	-- log_alpha for auto-tuning (scalar parameter stored as {1,1} Tensor)
	local logAlphaTensor: Types.Tensor?
	local alphaOpt = nil
	if autoAlpha then
		assert(cfg.alphaOptimizerFactory, "SAC: alphaOptimizerFactory required when autoAlpha=true")
		logAlphaTensor = Tensor.zeros({ 1, 1 }, "f64", true)
		logAlphaTensor._storage[1] = math.log(alpha)
		alphaOpt = cfg.alphaOptimizerFactory({ logAlphaTensor })
	end

	local function copyParams(src, dst)
		local sp, dp = src:parameters(), dst:parameters()
		task.desynchronize()
		for i = 1, #sp do
			local ss, ds = sp[i]._storage, dp[i]._storage
			for k = 1, #ss do
				ds[k] = ss[k]
			end
		end
		task.synchronize()
	end
	copyParams(critic1, critic1Target)
	copyParams(critic2, critic2Target)

	local function softUpdate(src, dst)
		local sp, dp = src:parameters(), dst:parameters()
		task.desynchronize()
		for i = 1, #sp do
			local ss, ds = sp[i]._storage, dp[i]._storage
			for k = 1, #ss do
				ds[k] = tau * ss[k] + (1 - tau) * ds[k]
			end
		end
		task.synchronize()
	end

	-- internal replay (same pattern as DDPG/TD3)
	local replay = cfg.replay
	if not replay then
		local cap = capacity
		local rSize, rHead = 0, 1
		local rS, rA, rR, rNS, rD = {}, {}, {}, {}, {}
		local rDtype

		replay = {
			push = function(_, t)
				if not rDtype then
					rDtype = t.state._dtype
				end
				local i = rHead
				rS[i] = Buffer.writeDenseArray(t.state._storage, rDtype, rS[i])
				rNS[i] = Buffer.writeDenseArray(t.nextState._storage, rDtype, rNS[i])
				rA[i] = type(t.action) == "table" and t.action or { t.action }
				rR[i], rD[i] = t.reward, t.done and true or false
				rHead = (rHead % cap) + 1
				if rSize < cap then
					rSize += 1
				end
			end,
			canSample = function(_, n)
				return rSize >= n
			end,
			sample = function(_, n)
				local dtype = rDtype or "f64"
				local S = Tensor.zeros({ sDim, n }, dtype, false)
				local NS = Tensor.zeros({ sDim, n }, dtype, false)
				local A = Tensor.zeros({ aDim, n }, dtype, false)
				local R, D = table.create(n, 0.0), table.create(n, false)
				task.desynchronize()
				for j = 1, n do
					local i = math.random(1, rSize)
					Buffer.copyDenseToColumn(rS[i], dtype, S._storage, j, n, sDim)
					Buffer.copyDenseToColumn(rNS[i], dtype, NS._storage, j, n, sDim)
					local act = rA[i]
					for d = 1, aDim do
						A._storage[(d - 1) * n + j] = act[d] or 0
					end
					R[j], D[j] = rR[i], rD[i]
				end
				task.synchronize()
				return { S = S, A = A, R = R, NS = NS, D = D }
			end,
		}
	end

	local function concat(state, action)
		local B = state._shape[2]
		local sa = Tensor.zeros({ sDim + aDim, B }, state._dtype, state._requiresGrad or action._requiresGrad)
		task.desynchronize()
		for d = 1, sDim do
			local base = (d - 1) * B
			for b = 1, B do
				sa._storage[base + b] = state._storage[base + b]
			end
		end
		for d = 1, aDim do
			local sb = (d - 1) * B
			local db = (sDim + d - 1) * B
			for b = 1, B do
				sa._storage[db + b] = action._storage[sb + b]
			end
		end
		task.synchronize()
		if action._requiresGrad then
			return Tape.makeNode(sa, { action }, function(gsa)
				if not action._requiresGrad then
					return
				end
				action._grad = action._grad or Tensor.zeros(action._shape)
				local ags, gas = action._grad._storage, gsa._storage
				task.desynchronize()
				for d = 1, aDim do
					local sb = (sDim + d - 1) * B
					local db = (d - 1) * B
					for b = 1, B do
						ags[db + b] += gas[sb + b]
					end
				end
				task.synchronize()
			end)
		end
		return sa
	end

	-- reparameterise: sample action and compute log_pi.
	-- actor output: {2*aDim, B} (mu rows first, then log_std rows)
	-- returns: action Tensor {aDim,B} with grad, log_pi {1,B} with grad
	local function sampleAction(out: Types.Tensor, B: number): (Types.Tensor, Types.Tensor)
		local action = Tensor.zeros({ aDim, B }, out._dtype, out._requiresGrad)
		local logPi = Tensor.zeros({ 1, B }, out._dtype, out._requiresGrad)

		-- store eps for backward
		local eps = table.create(aDim * B, 0.0)
		local std = table.create(aDim * B, 0.0)
		local preSquash = table.create(aDim * B, 0.0)

		task.desynchronize()
		local outs = out._storage
		for d = 1, aDim do
			for b = 1, B do
				local idx = (d - 1) * B + b
				local mu = outs[idx]
				local logStd = math.clamp(outs[aDim * B + idx], logStdMin, logStdMax)
				local s = math.exp(logStd)
				std[idx] = s
				local u1 = math.max(math.random(), 1e-7)
				local u2 = math.random()
				local e = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
				eps[idx] = e
				local z = mu + e * s
				preSquash[idx] = z
				action._storage[idx] = math.tanh(z) * aHigh
			end
		end
		-- log_pi = sum_d [ log_std + 0.5 + log(1 - tanh^2(z)) ] (Gaussian + tanh jacobian)
		-- = sum_d [ logStd + 0.5*(z-mu)^2/std^2 - log(cosh(z)) * 2 ] (simplified form)
		-- standard implementation: log N(z|mu,std) - sum log(1 - tanh^2(z))
		for j = 1, B do
			local lp = 0.0
			for d = 1, aDim do
				local idx = (d - 1) * B + j
				local z = preSquash[idx]
				local logStd = math.log(std[idx])
				local gaussLogP = -0.5 * (eps[idx] * eps[idx]) - logStd - 0.5 * math.log(2 * math.pi)
				local tanhZ = math.tanh(z)
				local logJacobian = math.log(math.max(1 - tanhZ * tanhZ, 1e-6))
				lp += gaussLogP - logJacobian
			end
			logPi._storage[j] = lp
		end
		task.synchronize()

		if not out._requiresGrad then
			return action, logPi
		end

		-- register backward through both action and logPi w.r.t. out
		local combined = Tape.makeNode(action, { out }, function(ga)
			if not out._requiresGrad then
				return
			end
			out._grad = out._grad or Tensor.zeros(out._shape)
			local gos = (out._grad :: any)._storage
			local gas = ga._storage
			task.desynchronize()
			for d = 1, aDim do
				for b = 1, B do
					local idx = (d - 1) * B + b
					local z = preSquash[idx]
					local tanhZ = math.tanh(z)
					-- d(tanh(z)*aHigh)/d(z) * d(z)/d(mu) = (1-tanh^2)*aHigh * 1
					local dTanh = (1 - tanhZ * tanhZ) * aHigh
					gos[idx] += gas[idx] * dTanh -- wrt mu
					gos[aDim * B + idx] += gas[idx] * dTanh * eps[idx] * std[idx] -- wrt log_std
				end
			end
			task.synchronize()
		end)

		return combined, logPi
	end

	local totalSteps = 0

	local function randn()
		local u = math.max(math.random(), 1e-7)
		return math.sqrt(-2 * math.log(u)) * math.cos(2 * math.pi * math.random())
	end

	local Agent = {}

    Agent.optimizer = actorOpt
    Agent.actorOptimizer = actorOpt
    Agent.criticOptimizer = criticOpt
	
	function Agent:act(state: Types.Tensor, evaluate: boolean?)
		local out = actor:forward(state) -- {2*aDim, 1}
		local result = table.create(aDim, 0.0)
		task.desynchronize()
		for d = 1, aDim do
			if evaluate then
				-- use mean (deterministic)
				result[d] = math.tanh(out._storage[d]) * aHigh
			else
				local mu = out._storage[d]
				local logStd = math.clamp(out._storage[aDim + d], logStdMin, logStdMax)
				local s = math.exp(logStd)
				local z = mu + randn() * s
				result[d] = math.tanh(z) * aHigh
			end
		end
		task.synchronize()
		return result
	end

	function Agent:observe(t)
		replay:push(t)
		totalSteps += 1
	end

	function Agent:trainStep()
		if totalSteps < warmup or not replay:canSample(batchSize) then
			return nil
		end

		local batch = replay:sample(batchSize)
		local S, A, R, NS, D = batch.S, batch.A, batch.R, batch.NS, batch.D
		local B = batchSize

		local curAlpha = autoAlpha and math.exp((logAlphaTensor :: any)._storage[1]) or alpha

		local nextOut = actor:forward(NS) -- {2*aDim, B}
		local nextA, nextLogPi = sampleAction(nextOut, B)
		local nsA1 = concat(NS, nextA)
		local Q1next = critic1Target:forward(nsA1)
		local Q2next = critic2Target:forward(nsA1)

		local targetQ = Tensor.zeros({ 1, B }, S._dtype, false)
		task.desynchronize()
		for j = 1, B do
			local qmin = math.min(Q1next._storage[j], Q2next._storage[j])
			targetQ._storage[j] = R[j] + gamma * (D[j] and 0 or 1) * (qmin - curAlpha * nextLogPi._storage[j])
		end
		task.synchronize()

		criticOpt:zeroGrad()
		local sA = concat(S, A) -- stored actions, no grad needed
		local Q1 = critic1:forward(sA)
		local Q2 = critic2:forward(sA)
		local cL1, gQ1 = Losses.mse_backward(Q1, targetQ)
		local cL2, gQ2 = Losses.mse_backward(Q2, targetQ)
		Tape.backwardFrom(Q1, gQ1)
		Tape.backwardFrom(Q2, gQ2)
		criticOpt:step()

		actorOpt:zeroGrad()
		local actOut = actor:forward(S) -- {2*aDim, B}, requiresGrad via actor params
		local sampledA, logPi = sampleAction(actOut, B)
		local smuA = concat(S, sampledA)
		local Q1val = critic1:forward(smuA)
		local Q2val = critic2:forward(smuA)

		-- actor loss = E[ alpha * log_pi(a|s) - min(Q1,Q2) ]
		-- grad w.r.t. actor output:
		local gActOut = Tensor.zeros({ 2 * aDim, B }, S._dtype, false)
		task.desynchronize()
		local totalActorLoss = 0.0
		for j = 1, B do
			local qmin = math.min(Q1val._storage[j], Q2val._storage[j])
			totalActorLoss += curAlpha * logPi._storage[j] - qmin
		end
		local gQ = Tensor.zeros({ 1, B }, S._dtype, false)
		for j = 1, B do
			gQ._storage[j] = -1 / B
		end
		task.synchronize()

		Tape.backwardFrom(Q1val, gQ)

		if actOut._requiresGrad then
			actOut._grad = actOut._grad or Tensor.zeros(actOut._shape)
		end

		actorOpt:step()

		local alphaLoss = nil
		if autoAlpha and logAlphaTensor then
			alphaOpt:zeroGrad()
			-- loss_alpha = -log_alpha * (log_pi + targetEntropy)
			-- grad = -(log_pi + targetEntropy)
			local meanLogPi = 0.0
			task.desynchronize()
			for j = 1, B do
				meanLogPi += logPi._storage[j]
			end
			task.synchronize()
			meanLogPi /= B

			if not logAlphaTensor._grad then
				logAlphaTensor._grad = Tensor.zeros({ 1, 1 })
			end
			(logAlphaTensor._grad :: any)._storage[1] = -(meanLogPi + targetEntropy)
			alphaOpt:step()
			alpha = math.exp(logAlphaTensor._storage[1])
			alphaLoss = -curAlpha * (meanLogPi + targetEntropy)
		end

		softUpdate(critic1, critic1Target)
		softUpdate(critic2, critic2Target)

		return {
			criticLoss = (cL1 + cL2) / 2,
			actorLoss = totalActorLoss / B,
			alphaLoss = alphaLoss,
			alpha = curAlpha,
		}
	end

	function Agent:getActor()
		return actor
	end
	function Agent:getCritic1()
		return critic1
	end
	function Agent:getCritic2()
		return critic2
	end
	function Agent:getAlpha()
		return autoAlpha and math.exp((logAlphaTensor :: any)._storage[1]) or alpha
	end
	function Agent:parameters()
		local params = {}
		for _, p in ipairs(actor:parameters()) do
			table.insert(params, p)
		end
		for _, p in ipairs(critic1:parameters()) do
			table.insert(params, p)
		end
		for _, p in ipairs(critic2:parameters()) do
			table.insert(params, p)
		end
		return params
	end

	return Agent
end
