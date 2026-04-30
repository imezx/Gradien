--!native
--!optimize 2
local Tensor = require("../Tensor")
local Tape = require("../autograd/Tape")
local Buffer = require("../Util/Buffer")
local Util = require("../Util")
local Losses = require("../nn/Losses")
local assert = Util.Assert

-- TD3 - Twin Delayed Deep Deterministic Policy Gradient (Fujimoto et al. 2018)
return function(cfg)
	assert(cfg.stateDim, "TD3: cfg.stateDim required")
	assert(cfg.actionDim, "TD3: cfg.actionDim required")
	assert(cfg.actorFactory, "TD3: cfg.actorFactory required")
	assert(cfg.criticFactory, "TD3: cfg.criticFactory required")

	local sDim = cfg.stateDim
	local aDim = cfg.actionDim
	local aHigh = cfg.actionHigh or 1.0
	local gamma = cfg.gamma or 0.99
	local tau = cfg.tau or 0.005
	local batchSize = cfg.batchSize or 64
	local explNoise = cfg.explorationNoise or 0.1
	local targetNoise = cfg.targetNoise or 0.2
	local noiseClip = cfg.noiseClip or 0.5
	local policyDelay = cfg.policyDelay or 2
	local warmup = cfg.warmupSteps or 1000
	local capacity = cfg.capacity or 100000

	local actor = cfg.actorFactory()
	local actorTarget = cfg.actorFactory()
	local critic1 = cfg.criticFactory()
	local critic2 = cfg.criticFactory()
	local critic1Target = cfg.criticFactory()
	local critic2Target = cfg.criticFactory()

	local actorOpt, criticOpt
	if cfg.actorOptimizerFactory then
		assert(cfg.criticOptimizerFactory, "TD3: criticOptimizerFactory required")
		actorOpt = cfg.actorOptimizerFactory(actor:parameters())
		-- single optimizer for both critics' combined parameters
		local cparams = {}
		for _, p in ipairs(critic1:parameters()) do
			table.insert(cparams, p)
		end
		for _, p in ipairs(critic2:parameters()) do
			table.insert(cparams, p)
		end
		criticOpt = cfg.criticOptimizerFactory(cparams)
	else
		assert(cfg.optimizerFactory, "TD3: provide optimizerFactory or actor/criticOptimizerFactory")
		actorOpt = cfg.optimizerFactory(actor:parameters())
		local cparams = {}
		for _, p in ipairs(critic1:parameters()) do
			table.insert(cparams, p)
		end
		for _, p in ipairs(critic2:parameters()) do
			table.insert(cparams, p)
		end
		criticOpt = cfg.optimizerFactory(cparams)
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
	copyParams(actor, actorTarget)
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

	-- internal replay (same continuous-action pattern as DDPG)
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

	local function randn()
		local u = math.max(math.random(), 1e-7)
		return math.sqrt(-2 * math.log(u)) * math.cos(2 * math.pi * math.random())
	end

	local totalSteps = 0
	local criticSteps = 0

	local Agent = {}

    Agent.optimizer = actorOpt
    Agent.actorOptimizer = actorOpt
    Agent.criticOptimizer = criticOpt
	
	function Agent:act(state, evaluate)
		local a = actor:forward(state)
		local out = table.create(aDim, 0.0)
		task.desynchronize()
		for d = 1, aDim do
			local v = a._storage[d] * aHigh
			if not evaluate then
				v += randn() * explNoise
			end
			out[d] = math.clamp(v, -aHigh, aHigh)
		end
		task.synchronize()
		return out
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

		-- target action with clipped smoothing noise
		local nextA = actorTarget:forward(NS) -- {aDim, B}
		task.desynchronize()
		for k = 1, #nextA._storage do
			local n = math.clamp(randn() * targetNoise, -noiseClip, noiseClip)
			nextA._storage[k] = math.clamp(nextA._storage[k] * aHigh + n, -aHigh, aHigh)
		end
		task.synchronize()

		local nsA = concat(NS, nextA)
		local Q1next = critic1Target:forward(nsA)
		local Q2next = critic2Target:forward(nsA)

		local targetQ = Tensor.zeros({ 1, B }, S._dtype, false)
		task.desynchronize()
		for j = 1, B do
			local qmin = math.min(Q1next._storage[j], Q2next._storage[j])
			targetQ._storage[j] = R[j] + gamma * (D[j] and 0 or 1) * qmin
		end
		task.synchronize()

		criticOpt:zeroGrad()
		local sA = concat(S, A)
		local Q1 = critic1:forward(sA)
		local Q2 = critic2:forward(sA)
		local l1, g1 = Losses.mse_backward(Q1, targetQ)
		local l2, g2 = Losses.mse_backward(Q2, targetQ)
		Tape.backwardFrom(Q1, g1)
		Tape.backwardFrom(Q2, g2)
		criticOpt:step()
		criticSteps += 1

		local actorLoss = nil
		if criticSteps % policyDelay == 0 then
			actorOpt:zeroGrad()
			local muA = actor:forward(S)
			local smuA = concat(S, muA)
			local Qval = critic1:forward(smuA)
			local gA = Tensor.zeros({ 1, B }, S._dtype, false)
			task.desynchronize()
			for j = 1, B do
				gA._storage[j] = -1 / B
			end
			task.synchronize()
			Tape.backwardFrom(Qval, gA)
			actorOpt:step()

			softUpdate(actor, actorTarget)
			softUpdate(critic1, critic1Target)
			softUpdate(critic2, critic2Target)

			actorLoss = -Qval._storage[1] -- approx
		end

		return { criticLoss = (l1 + l2) / 2, actorLoss = actorLoss }
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
