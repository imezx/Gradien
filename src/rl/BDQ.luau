--!native
--!optimize 2
local G = script.Parent.Parent
local Tensor = require("../Tensor")
local Tape = require("../autograd/Tape")
local Losses = require("../nn/Losses")
local Util = require("../Util")
local assert = Util.Assert

local function makeActLayer(actAny: any)
	local A = G.NN.Activations
	if type(actAny) == "function" then
		return function(_, x) return actAny(x) end
	end
	local name = (actAny ~= nil and actAny) or "ReLU"
	local af = A[name]
	assert(type(af) == "function", ("BDQ: unknown activation '%s'"):format(tostring(name)))
	return function(_, x) return af(x) end
end

--@param D number @state dim
--@param H number @hidden width
--@param branches {number}
--@param act any @string or function
--@param depth number @trunk depth >=1
local function buildDefaultBranched(D: number, H: number, branches: {number}, act: any, depth: number)
	local actLayer = makeActLayer(act)
	local layers = { G.NN.Linear(D, H), actLayer }
	local V = G.NN.Linear(H, 1) -- state value head
	for _ = 2, depth do
		table.insert(layers, G.NN.Linear(H, H))
		table.insert(layers, actLayer)
	end
	local trunk = G.NN.Sequential(layers)

	local heads = {}
	for b, k in ipairs(branches) do
		heads[b] = G.NN.Linear(H, k)
	end

	local M = {}
	function M:forward(s)
		local z = trunk:forward(s) -- {H,B}
		local v = V:forward(z) -- {1,B}
		local outs = {}
		for b, k in ipairs(branches) do
			local A = heads[b]:forward(z) -- {k,B} advantages
			local Q = G.Tensor.zeros(A._shape)
			local Bsz = A._shape[2]
			task.desynchronize()
			for j=1,Bsz do
				local m=0; for i=1,k do m += A._storage[(i-1)*Bsz + j] end
				m /= k
				for i=1,k do
					Q._storage[(i-1)*Bsz + j] = v._storage[j] + (A._storage[(i-1)*Bsz + j] - m)
				end
			end
			task.synchronize()
			outs[b] = Q
		end
		return outs
	end
	function M:parameters()
		local p = trunk:parameters()
		for b = 1, #branches do
			local hp = heads[b]:parameters()
			for i = 1, #hp do table.insert(p, hp[i]) end
		end
		return p
	end
	-- init
	for _, p in M:parameters() do
		if p._shape and p._shape[2] then G.Initializers.kaimingUniform(p) else G.Initializers.zeros(p) end
	end
	return M
end

local function argmaxPerColumn(Q)
	local K, B = Q._shape[1], Q._shape[2]
	local idx = table.create(B, 1)
	task.desynchronize()
	for j = 1, B do
		local best, ai = -1/0, 1
		for i = 1, K do
			local v = Q._storage[(i - 1) * B + j]
			if v > best then best, ai = v, i end
		end
		idx[j] = ai
	end
	task.synchronize()
	return idx
end

local function gatherChosen(Q, A)
	local K, B = Q._shape[1], Q._shape[2]
	local v = Tensor.zeros({1, B})
	task.desynchronize()
	for j = 1, B do
		local ai = A[j]; if ai < 1 then ai = 1 elseif ai > K then ai = K end
		v._storage[j] = Q._storage[(ai - 1) * B + j]
	end
	task.synchronize()
	return v
end

local function sumHeads(values: {any})
	local B = values[1]._shape[2]
	local s = Tensor.zeros({1, B})
	task.desynchronize()
	for j = 1, B do
		local acc = 0.0
		for b = 1, #values do
			acc += values[b]._storage[j]
		end
		s._storage[j] = acc
	end
	task.synchronize()
	return s
end

--@param Qheads {any} -- list of Q^b{Kb,B}
--@param actions {{number}} -- list per head: A^b{B}
--@param gsum any -- {1,B} gradient wrt sum
--@return {any} -- grads per head (same shapes as Q^b)
local function scatterGrads(Qheads, actions, gsum)
	local B = gsum._shape[2]
	local grads = table.create(#Qheads)
	task.desynchronize()
	for b = 1, #Qheads do
		local Qb, Ab = Qheads[b], actions[b]
		local Gb = Tensor.zeros(Qb._shape)
		for j = 1, B do
			local ai = Ab[j]
			Gb._storage[(ai - 1) * B + j] = gsum._storage[j]
		end
		grads[b] = Gb
	end
	task.synchronize()
	return grads
end

return function(cfg)
	assert(cfg.modelFactory or (cfg.stateDim and cfg.hidden and cfg.branches),
		"BDQ: provide modelFactory OR (stateDim, hidden, branches)")
	local branches = cfg.branches
	local policy = cfg.modelFactory and cfg.modelFactory() or buildDefaultBranched(
		cfg.stateDim :: number,
		cfg.hidden :: number,
		branches :: {number},
		cfg.activation,
		(cfg.depth or 2) :: number
	)
	local target = cfg.modelFactory and cfg.modelFactory() or buildDefaultBranched(
		cfg.stateDim :: number,
		cfg.hidden :: number,
		branches :: {number},
		cfg.activation,
		(cfg.depth or 2) :: number
	)

	local function hardSync()
		local fps, tps = policy:parameters(), target:parameters()
		task.desynchronize()
		for i = 1, #fps do
			for k = 1, #fps[i]._storage do
				tps[i]._storage[k] = fps[i]._storage[k]
			end
		end
		task.synchronize()
	end
	hardSync()

	local tau = cfg.tau
	local function softSync()
		if not tau then return end
		local fps, tps = policy:parameters(), target:parameters()
		task.desynchronize()
		for i = 1, #fps do
			for k = 1, #fps[i]._storage do
				local tv = tps[i]._storage[k]
				local pv = fps[i]._storage[k]
				tps[i]._storage[k] = (1 - tau) * tv + tau * pv
			end
		end
		task.synchronize()
	end

	local opt = cfg.optimizerFactory(policy:parameters())

	local gamma = cfg.gamma or 0.99
	local batchSize = cfg.batchSize or 1
	local lossKind = (cfg.loss == "mse") and "mse" or "huber"
	local huberDelta = cfg.huberDelta or 1.0

	local epsStart = cfg.epsilonStart or 1.0
	local epsEnd = cfg.epsilonEnd or 0.10
	local epsDecay = cfg.epsilonDecay or 30000

	local stepCounter = 0
	local lastTransition

	local function epsAt(n)
		local k = math.exp(-n / epsDecay)
		return epsEnd + (epsStart - epsEnd) * k
	end

	local Agent = {}

	Agent.getPolicy = function(_)
		return policy
	end

	Agent.parameters = function(_)
		return policy:parameters()
	end
	
	--@param state any -- {D,1} or {D,B}
	--@param n number -- global step (for epsilon schedule)
	--@return {number}
	function Agent:act(state, n)
		local B = state._shape[2]
		assert(B == 1, "BDQ:act expects {D,1} in online use; got batch "..tostring(B))
		local eps = epsAt(n)
		if math.random() < eps then
			local a = {}
			for b = 1, #branches do a[b] = math.random(1, branches[b]) end
			return a
		end
		local Qh = policy:forward(state)
		local a = {}
		for b = 1, #Qh do
			local idx = argmaxPerColumn(Qh[b]) -- length 1
			a[b] = idx[1]
		end
		return a
	end

	--@param t {state:any, actions:{number}, reward:number, nextState:any, done:boolean?}
	function Agent:observe(t)
		if cfg.replay then
			cfg.replay:push(t)
		else
			lastTransition = t
		end
	end

	function Agent:trainStep()
		local batch
		if cfg.replay then
			if not cfg.replay:canSample(batchSize) then return nil end
			batch = cfg.replay:sample(batchSize)
		else
			if not lastTransition then return nil end
			batch = {
				S = lastTransition.state,
				NS = lastTransition.nextState,
				R = { lastTransition.reward },
				D = { lastTransition.done and true or false },
				indices = nil,
				weights = nil,
				actions = lastTransition.actions, -- { per-head index } for B=1
			}
		end

		local S, NS, R, D = batch.S, batch.NS, batch.R, batch.D
		local B = (S._shape and S._shape[2]) or 1
		local A = {}
		if batch.actions then
			for b = 1, #branches do
				A[b] = { batch.actions[b] }
			end
		else
			if batch.A and type(batch.A[1]) == "table" then
				A = batch.A
			elseif batch.A and type(batch.A[1]) == "number" then
				error("BDQ: expected replay to store per-head actions (A[b][j])")
			else
				error("BDQ: missing actions in batch")
			end
		end

		local W = batch.weights

		local Qh = policy:forward(S) -- list of {K_b,B}
		local chosenPerHead = {}
		for b = 1, #Qh do
			chosenPerHead[b] = gatherChosen(Qh[b], A[b]) -- {1,B}
		end
		local Qsum = sumHeads(chosenPerHead) -- {1,B}

		local Qh_ns_pol = policy:forward(NS)
		local Astar = {}
		for b = 1, #Qh_ns_pol do
			Astar[b] = argmaxPerColumn(Qh_ns_pol[b]) -- {B}
		end
		local Qh_ns_tgt = target:forward(NS)
		local chosenNext = {}
		for b = 1, #Qh_ns_tgt do
			chosenNext[b] = gatherChosen(Qh_ns_tgt[b], Astar[b]) -- {1,B}
		end
		local QnextSum = sumHeads(chosenNext) -- {1,B}

		local Y = Tensor.zeros({1, B})
		task.desynchronize()
		for j = 1, B do
			local r = R[j] or 0
			local done = D[j] and 1 or 0
			local qn = QnextSum._storage[j]
			Y._storage[j] = r + (1 - done) * (gamma * qn)
		end
		task.synchronize()

		local loss, dQsum
		if lossKind == "mse" then
			loss, dQsum = Losses.mse_backward(Qsum, Y)
		else
			loss, dQsum = Losses.huber_backward(Qsum, Y, huberDelta)
		end

		if W then
			task.desynchronize()
			for j = 1, B do dQsum._storage[j] *= (W[j] or 1.0) end
			task.synchronize()
		end

		local headGrads = scatterGrads(Qh, A, dQsum)
		opt:zeroGrad()
		for b = 1, #Qh do
			Tape.backwardFrom(Qh[b], headGrads[b])
		end
		opt:step()

		stepCounter += 1
		
		if tau then
			softSync()
		elseif cfg.targetSyncInterval and stepCounter % cfg.targetSyncInterval == 0 then
			hardSync()
		end

		if cfg.replay and cfg.replay.updatePriorities and batch.indices then
			local td = table.create(B, 0.0)
			task.desynchronize()
			for j = 1, B do
				td[j] = math.abs(Qsum._storage[j] - Y._storage[j])
			end
			task.synchronize()
			cfg.replay:updatePriorities(batch.indices, td)
		end

		return loss
	end
	
	Agent.loadParameters = function(_, snapshot, strict)
		local fp = policy:parameters()
		if strict ~= false then
			assert(#fp == #snapshot, "snapshot size mismatch")
			for i=1,#fp do
				assert(#fp[i]._storage == #snapshot[i], ("tensor size mismatch at %d, got %d:%d"):format(i, #fp[i]._storage, #snapshot[i]))
			end
		end
		local n = math.min(#fp, #snapshot)
		task.desynchronize()
		for i=1,n do
			local p, s = fp[i], snapshot[i]
			local m = math.min(#p._storage, #s)
			for k=1,m do p._storage[k] = s[k] end
		end
		task.synchronize()
		-- refresh target
		local fp2, tp = policy:parameters(), target:parameters()
		task.desynchronize()
		for i=1,#fp2 do
			for k=1,#fp2[i]._storage do tp[i]._storage[k] = fp2[i]._storage[k] end
		end
		task.synchronize()
	end
	return Agent
end
