--!native
--!optimize 2
local Tensor = require("../Tensor")
local Tape = require("../autograd/Tape")
local Buffer = require("../Util/Buffer")
local Util = require("../Util")
local Types = require("../Types")
local Losses = require("../nn/Losses")
local assert = Util.Assert

-- DDPG - Deep Deterministic Policy Gradient (Lillicrap et al. 2015)
return function(cfg)
	assert(cfg.stateDim, "DDPG: cfg.stateDim required")
	assert(cfg.actionDim, "DDPG: cfg.actionDim required")
	assert(cfg.actorFactory, "DDPG: cfg.actorFactory required")
	assert(cfg.criticFactory, "DDPG: cfg.criticFactory required")

	local sDim = cfg.stateDim
	local aDim = cfg.actionDim
	local aHigh = cfg.actionHigh or 1.0
	local gamma = cfg.gamma or 0.99
	local tau = cfg.tau or 0.005
	local batchSize = cfg.batchSize or 64
	local explNoise = cfg.explorationNoise or 0.1
	local warmup = cfg.warmupSteps or 1000
	local capacity = cfg.capacity or 100000

	-- networks
	local actor = cfg.actorFactory()
	local actorTarget = cfg.actorFactory()
	local critic = cfg.criticFactory()
	local criticTarget = cfg.criticFactory()

	-- optimizers
	local actorOpt, criticOpt
	if cfg.actorOptimizerFactory then
		assert(cfg.criticOptimizerFactory, "DDPG: criticOptimizerFactory required with actorOptimizerFactory")
		actorOpt = cfg.actorOptimizerFactory(actor:parameters())
		criticOpt = cfg.criticOptimizerFactory(critic:parameters())
	else
		assert(cfg.optimizerFactory, "DDPG: provide optimizerFactory or actor/criticOptimizerFactory")
		actorOpt = cfg.optimizerFactory(actor:parameters())
		criticOpt = cfg.optimizerFactory(critic:parameters())
	end

	-- hard-copy params src -> dst
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
	-- initialise target networks as copies of online networks
	copyParams(actor, actorTarget)
	copyParams(critic, criticTarget)

	-- soft update: dst = tau*src + (1-tau)*dst
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

	-- internal replay buffer (stores continuous actions as dense arrays)
	local replay = cfg.replay
	if not replay then
		local cap = capacity
		local rSize, rHead = 0, 1
		local rS, rA, rR, rNS, rD = {}, {}, {}, {}, {}
		local rDtype

		local function rpush(t)
			assert(t.state and t.nextState and t.action, "DDPG replay: state/action/nextState required")
			if not rDtype then
				rDtype = t.state._dtype
			end
			local i = rHead
			rS[i] = Buffer.writeDenseArray(t.state._storage, rDtype, rS[i])
			rNS[i] = Buffer.writeDenseArray(t.nextState._storage, rDtype, rNS[i])
			-- store continuous action as a plain array
			if type(t.action) == "table" then
				rA[i] = t.action
			else
				-- single-dim action wrapped in table
				rA[i] = { t.action }
			end
			rR[i], rD[i] = t.reward, t.done and true or false
			rHead = (rHead % cap) + 1
			if rSize < cap then
				rSize += 1
			end
		end

		local function rcanSample(n)
			return rSize >= n
		end

		local function rsample(n)
			local dtype = rDtype or "f64"
			local S = Tensor.zeros({ sDim, n }, dtype, false)
			local NS = Tensor.zeros({ sDim, n }, dtype, false)
			local A = Tensor.zeros({ aDim, n }, dtype, false)
			local R = table.create(n, 0.0)
			local D = table.create(n, false)
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
		end

		replay = { push = rpush, canSample = rcanSample, sample = rsample }
	end

	-- concatenate state {sDim,B} and action {aDim,B} -> {sDim+aDim,B}
	local function concat(state: Types.Tensor, action: Types.Tensor): Types.Tensor
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
			local srcBase = (d - 1) * B
			local dstBase = (sDim + d - 1) * B
			for b = 1, B do
				sa._storage[dstBase + b] = action._storage[srcBase + b]
			end
		end
		task.synchronize()
		-- grad routing: we only need grad w.r.t. action for the actor loss
		if action._requiresGrad then
			return Tape.makeNode(sa, { action }, function(gsa)
				if not action._requiresGrad then
					return
				end
				action._grad = action._grad or Tensor.zeros(action._shape)
				local ags = (action._grad :: any)._storage
				local gas = gsa._storage
				task.desynchronize()
				for d = 1, aDim do
					local srcBase = (sDim + d - 1) * B
					local dstBase = (d - 1) * B
					for b = 1, B do
						ags[dstBase + b] += gas[srcBase + b]
					end
				end
				task.synchronize()
			end)
		end
		return sa
	end

	local totalSteps = 0

	-- randn helper
	local function randn()
		local u = math.max(math.random(), 1e-7)
		return math.sqrt(-2 * math.log(u)) * math.cos(2 * math.pi * math.random())
	end

	local Agent = {}

    Agent.optimizer = actorOpt
    Agent.actorOptimizer = actorOpt
    Agent.criticOptimizer = criticOpt

	-- returns a {number} array of length actionDim.
	function Agent:act(state: Types.Tensor, evaluate: boolean?)
		local a = actor:forward(state) -- {aDim, 1}
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

		-- y = r + gamma * (1-done) * Q'(s', mu'(s'))
		local nextA = actorTarget:forward(NS) -- {aDim, B}
		local nsA = concat(NS, nextA)
		local Qnext = criticTarget:forward(nsA) -- {1, B}

		local targetQ = Tensor.zeros({ 1, B }, S._dtype, false)
		task.desynchronize()
		for j = 1, B do
			local cont = D[j] and 0 or 1
			targetQ._storage[j] = R[j] + gamma * cont * Qnext._storage[j]
		end
		task.synchronize()

		criticOpt:zeroGrad()
		local sA = concat(S, A) -- A has no grad here
		local Q = critic:forward(sA) -- {1, B}
		local criticLoss, gQ = Losses.mse_backward(Q, targetQ)
		Tape.backwardFrom(Q, gQ)
		criticOpt:step()

		-- maximize Q(s, mu(s)) -> minimize -Q
		actorOpt:zeroGrad()
		local muA = actor:forward(S) -- {aDim, B}, requiresGrad=true via actor params
		local smuA = concat(S, muA) -- grad flows through muA
		local Qval = critic:forward(smuA) -- {1, B}

		-- actor loss grad: -dQ/da * da/d(actor params)
		local gActor = Tensor.zeros({ 1, B }, S._dtype, false)
		task.desynchronize()
		for j = 1, B do
			gActor._storage[j] = -1 / B
		end
		task.synchronize()
		Tape.backwardFrom(Qval, gActor)
		actorOpt:step()

		-- soft update targets
		softUpdate(actor, actorTarget)
		softUpdate(critic, criticTarget)

		return { criticLoss = criticLoss }
	end

	function Agent:getActor()
		return actor
	end
	function Agent:getCritic()
		return critic
	end
	function Agent:parameters()
		local params = {}
		for _, p in ipairs(actor:parameters()) do
			table.insert(params, p)
		end
		for _, p in ipairs(critic:parameters()) do
			table.insert(params, p)
		end
		return params
	end

	return Agent
end
