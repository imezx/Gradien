--!optimize 2
local Tensor = require("../Tensor")
local Tape = require("../autograd/Tape")
local Losses = require("../nn/Losses")
local Softmax = require("../nn/Softmax")

return function(cfg)
	assert(cfg.policy, "PPO: cfg.policy (policy network) required")
	assert(cfg.value, "PPO: cfg.value (value network) required")

	local policy = cfg.policy
	local value = cfg.value
	local gamma = cfg.gamma or 0.99
	local lam = cfg.lam or 0.95
	local clipEps = cfg.clip or 0.2
	local minBatch = cfg.minBatch or 32
	local epochs = cfg.epochs or 4
	local maxBuffer = cfg.maxBuffer or (minBatch * 4)

	local policyOpt, valueOpt

	if cfg.policyOptimizerFactory or cfg.valueOptimizerFactory then
		assert(cfg.policyOptimizerFactory, "PPO: cfg.policyOptimizerFactory required when using separate optimizers")
		assert(cfg.valueOptimizerFactory, "PPO: cfg.valueOptimizerFactory required when using separate optimizers")
		policyOpt = cfg.policyOptimizerFactory(policy:parameters())
		valueOpt = cfg.valueOptimizerFactory(value:parameters())
	else
		assert(
			cfg.optimizerFactory,
			"PPO: provide optimizerFactory or (policyOptimizerFactory + valueOptimizerFactory)"
		)
		local p = policy:parameters()
		local v = value:parameters()
		policyOpt = cfg.optimizerFactory(p)
		valueOpt = cfg.optimizerFactory(v)
	end

	local S_buf, NS_buf, A_buf, R_buf, D_buf, logp_buf = {}, {}, {}, {}, {}, {}

	local function clearBuf()
		table.clear(S_buf)
		table.clear(NS_buf)
		table.clear(A_buf)
		table.clear(R_buf)
		table.clear(D_buf)
		table.clear(logp_buf)
	end

	local function maybeTrim()
		local n = #S_buf
		if n <= maxBuffer then
			return
		end
		local drop = n - maxBuffer
		task.desynchronize()
		for i = 1, drop do
			table.remove(S_buf, 1)
			table.remove(NS_buf, 1)
			table.remove(A_buf, 1)
			table.remove(R_buf, 1)
			table.remove(D_buf, 1)
			table.remove(logp_buf, 1)
		end
		task.synchronize()
	end

	local function packStates(states)
		local T = #states
		assert(T > 0, "PPO: empty state buffer")
		local s0 = states[1]
		local sd = s0._shape[1]
		local B = T
		local S = Tensor.zeros({ sd, B }, s0._dtype, true)

		task.desynchronize()
		for j = 1, B do
			local s = states[j]
			assert(s._shape[1] == sd and s._shape[2] == 1, "PPO: expected state {D,1}")
			for i = 1, sd do
				S._storage[(i - 1) * B + j] = s._storage[i]
			end
		end
		task.synchronize()
		return S
	end

	local function computeGAE()
		local T = #R_buf
		assert(T > 0, "PPO: empty reward buffer")

		local S = packStates(S_buf)
		local NS = packStates(NS_buf)
		local B = T

		local V = value:forward(S) -- {1,B}
		local Vn = value:forward(NS) -- {1,B}

		local advantages = table.create(B, 0.0)
		local returns = table.create(B, 0.0)

		task.desynchronize()
		local deltas = table.create(B, 0.0)
		for t = 1, B do
			local r = R_buf[t] or 0.0
			local done = D_buf[t] and 1 or 0
			local vt = V._storage[t]
			local vtn = done == 1 and 0.0 or Vn._storage[t]
			local delta = r + gamma * vtn - vt
			deltas[t] = delta
		end

		local adv = 0.0
		for t = B, 1, -1 do
			local done = D_buf[t] and 1 or 0
			adv = deltas[t] + gamma * lam * (1 - done) * adv
			advantages[t] = adv
			local vt = V._storage[t]
			returns[t] = adv + vt
		end

		local mean, var = 0.0, 0.0
		for t = 1, B do
			mean += advantages[t]
		end
		mean /= B
		for t = 1, B do
			local d = advantages[t] - mean
			var += d * d
		end
		var = var / math.max(B, 1)
		local std = math.sqrt(var) + 1e-8
		for t = 1, B do
			advantages[t] = (advantages[t] - mean) / std
		end
		task.synchronize()

		return {
			S = S,
			V = V,
			advantages = advantages,
			returns = returns,
		}
	end

	local function trainFromBuffer()
		local T = #S_buf
		if T < minBatch then
			return nil
		end

		local info = computeGAE()
		local S = info.S
		local V = info.V
		local advantages = info.advantages
		local returns = info.returns

		local B = #advantages

		local target = Tensor.zeros({ 1, B }, V._dtype, false)
		task.desynchronize()
		for j = 1, B do
			target._storage[j] = returns[j]
		end
		task.synchronize()

		valueOpt:zeroGrad()
		local vLoss, gV = Losses.mse_backward(V, target)
		Tape.backwardFrom(V, gV)
		valueOpt:step()

		local lastPolicyLoss = 0.0

		for _ = 1, epochs do
			policyOpt:zeroGrad()
			local logits = policy:forward(S) -- {A,B}
			local P = Softmax.forward(logits)
			local C = logits._shape[1]

			local grad = Tensor.zeros(logits._shape, logits._dtype, false)
			local lossSum = 0.0

			task.desynchronize()
			for j = 1, B do
				local a = A_buf[j]
				local idx = (a - 1) * B + j
				local prob = P._storage[idx]
				local logp = math.log(math.max(prob, 1e-12))
				local logp_old = logp_buf[j]
				local ratio = math.exp(logp - logp_old)
				local clipped = math.clamp(ratio, 1 - clipEps, 1 + clipEps)

				local adv = advantages[j]
				local w = clipped * adv

				lossSum += -w * logp

				for i = 1, C do
					local pij = P._storage[(i - 1) * B + j]
					local indicator = (i == a) and 1 or 0
					local g = -(w / B) * (indicator - pij)
					grad._storage[(i - 1) * B + j] += g
				end
			end
			task.synchronize()

			lastPolicyLoss = lossSum / B

			Tape.backwardFrom(logits, grad)
			policyOpt:step()
		end

		local avgReturn = 0.0
		for j = 1, B do
			avgReturn += returns[j]
		end
		avgReturn /= B

		clearBuf()
		return {
			policyLoss = lastPolicyLoss,
			valueLoss = vLoss,
			avgReturn = avgReturn,
			batchSize = B,
		}
	end

	local Agent = {}

	function Agent:getPolicy()
		return policy
	end

	function Agent:getValue()
		return value
	end

	function Agent:parameters()
		local params = {}
		local pp = policy:parameters()
		local vp = value:parameters()
		for i = 1, #pp do
			table.insert(params, pp[i])
		end
		for i = 1, #vp do
			table.insert(params, vp[i])
		end
		return params
	end

	function Agent:act(state)
		local logits = policy:forward(state) -- {A,1}
		local P = Softmax.forward(logits)
		local B = logits._shape[2]
		assert(B == 1, "PPO: act() expects state with batch=1")
		local A = logits._shape[1]

		task.desynchronize()
		local r = math.random()
		local cum = 0.0
		local chosen = A
		for i = 1, A do
			cum += P._storage[i]
			if r <= cum then
				chosen = i
				break
			end
		end
		task.synchronize()
		return chosen
	end

	function Agent:observe(t)
		assert(t.state and t.nextState and t.action, "PPO.observe: expected {state, action, reward, nextState, done}")
		table.insert(S_buf, t.state)
		table.insert(NS_buf, t.nextState)
		table.insert(A_buf, t.action)
		table.insert(R_buf, t.reward or 0.0)
		table.insert(D_buf, t.done and true or false)

		local logits = policy:forward(t.state)
		local P = Softmax.forward(logits)
		local B = logits._shape[2]
		assert(B == 1, "PPO.observe: state must have batch=1")
		local a = t.action
		local idx = (a - 1) * B + 1

		task.desynchronize()
		local prob = P._storage[idx]
		local logp = math.log(math.max(prob, 1e-12))
		table.insert(logp_buf, logp)
		task.synchronize()

		maybeTrim()
	end

	function Agent:trainStep()
		return trainFromBuffer()
	end

	function Agent:resetBuffer()
		clearBuf()
	end

	return Agent
end
