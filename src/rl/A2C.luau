--!native
--!optimize 2
local Tensor = require("../Tensor")
local Tape = require("../autograd/Tape")
local Losses = require("../nn/Losses")
local Softmax = require("../nn/Softmax")
local Util = require("../Util")
local assert = Util.Assert

return function(cfg)
	local gamma = cfg.gamma or 0.99
	local minBatch = cfg.minBatch or 8

	local policy = assert(cfg.policy, "A2C: cfg.policy (actor network) required")
	local value = assert(cfg.value, "A2C: cfg.value (critic network) required")

	local policyOpt
	local valueOpt

	if cfg.policyOptimizerFactory or cfg.valueOptimizerFactory then
		assert(cfg.policyOptimizerFactory, "A2C: cfg.policyOptimizerFactory required when using separate optimizers")
		assert(cfg.valueOptimizerFactory, "A2C: cfg.valueOptimizerFactory required when using separate optimizers")
		policyOpt = cfg.policyOptimizerFactory(policy:parameters())
		valueOpt = cfg.valueOptimizerFactory(value:parameters())
	else
		assert(
			cfg.optimizerFactory,
			"A2C: provide optimizerFactory or (policyOptimizerFactory + valueOptimizerFactory)"
		)
		policyOpt = cfg.optimizerFactory(policy:parameters())
		valueOpt = cfg.optimizerFactory(value:parameters())
	end

	local S_buf = {}
	local A_buf = {}
	local R_buf = {}
	local D_buf = {}

	local function clearBuf()
		table.clear(S_buf)
		table.clear(A_buf)
		table.clear(R_buf)
		table.clear(D_buf)
	end

	local function packStates()
		local T = #S_buf
		assert(T > 0, "A2C: empty buffer")
		local s0 = S_buf[1]
		local sd = s0._shape[1]
		local B = T
		local S = Tensor.zeros({ sd, B }, true)

		task.desynchronize()
		for j = 1, B do
			local s = S_buf[j]
			assert(s._shape[1] == sd and s._shape[2] == 1, "A2C: expected state {D,1}")
			for i = 1, sd do
				S._storage[(i - 1) * B + j] = s._storage[i]
			end
		end
		task.synchronize()
		return S
	end

	local function computeReturns()
		local B = #R_buf
		local G = table.create(B, 0.0)
		local ret = 0.0
		task.desynchronize()
		for t = B, 1, -1 do
			if D_buf[t] then
				ret = 0.0
			end
			ret = R_buf[t] + gamma * ret
			G[t] = ret
		end
		task.synchronize()
		return G
	end

	local function trainFromBuffer()
		local T = #S_buf
		if T == 0 then
			return nil
		end

		local S = packStates()
		local B = T

		-- critic update: V(s) -> returns
		local V = value:forward(S) -- {1,B}
		local G = computeReturns()
		local target = Tensor.zeros({ 1, B }, false)

		task.desynchronize()
		for j = 1, B do
			target._storage[j] = G[j]
		end
		task.synchronize()

		valueOpt:zeroGrad()
		local vLoss, gV = Losses.mse_backward(V, target)
		Tape.backwardFrom(V, gV)
		valueOpt:step()

		-- compute advantages A_t = G_t - V(s_t)
		local advantages = table.create(B, 0.0)
		task.desynchronize()
		for j = 1, B do
			local vj = V._storage[j]
			advantages[j] = G[j] - vj
		end
		task.synchronize()

		-- actor update with policy gradient
		policyOpt:zeroGrad()
		local logits = policy:forward(S) -- {A,B}
		local C = logits._shape[1]
		local P = Softmax.forward(logits)
		local grad = Tensor.zeros(logits._shape, false)

		local lossSum = 0.0
		task.desynchronize()
		for j = 1, B do
			local a = A_buf[j]
			local adv = advantages[j]
			local idx = (a - 1) * B + j
			local prob = P._storage[idx]
			local logp = math.log(math.max(prob, 1e-12))
			lossSum += -adv * logp

			for i = 1, C do
				local pij = P._storage[(i - 1) * B + j]
				local indicator = (i == a) and 1 or 0
				-- dL/dz_i = A * (p_i - 1_{i=a})
				grad._storage[(i - 1) * B + j] = adv * (pij - indicator)
			end
		end
		for k = 1, #grad._storage do
			grad._storage[k] /= B
		end
		task.synchronize()

		local pLoss = lossSum / B

		Tape.backwardFrom(logits, grad)
		policyOpt:step()

		clearBuf()
		return {
			policyLoss = pLoss,
			valueLoss = vLoss,
		}
	end

	local Agent = {}

    Agent.optimizer = policyOpt
    Agent.policyOptimizer = policyOpt
    Agent.valueOptimizer = valueOpt
	Agent.policy = policy

	--@param state {D,1}
	--@return number
	function Agent:act(state)
		local logits = policy:forward(state)
		local B = logits._shape[2]
		assert(B == 1, "A2C: act() expects state with batch=1")
		local C = logits._shape[1]
		local P = Softmax.forward(logits)

		local r = math.random()
		local cum = 0.0
		-- j = 1 column, so index = i
		for i = 1, C do
			cum += P._storage[i]
			if r <= cum then
				return i
			end
		end
		return C
	end

	--@param t { state = Tensor, action = number, reward = number, done = boolean }
	function Agent:observe(t)
		table.insert(S_buf, t.state)
		table.insert(A_buf, t.action)
		table.insert(R_buf, t.reward)
		table.insert(D_buf, t.done == true)
	end

	--@return {policyLoss:number, valueLoss:number} or nil
	function Agent:trainStep()
		if #S_buf < minBatch then
			return nil
		end
		return trainFromBuffer()
	end

	--@return nn.Module
	function Agent:getPolicy()
		return policy
	end

	--@return nn.Module
	function Agent:getValue()
		return value
	end

	--@return {Tensor}
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

	function Agent:resetBuffer()
		clearBuf()
	end

	return Agent
end
