--!native
--!optimize 2
local Types = require("../../Types")
local Tensor = require("../../Tensor")
local Linear = require("../../nn/Linear")
local LSTM = require("../../nn/LSTM")
local Softmax = require("../../nn/Softmax")
local Util = require("../../Util")
local Activations = require("../../nn/Activations")
local assert = Util.Assert

local function cosineSimilarity(u: Types.Tensor, v: Types.Tensor)
	-- u, v: {D, B}
	local dot = Tensor.zeros({ 1, u._shape[2] }, u._dtype, u._requiresGrad or v._requiresGrad)
	local B = u._shape[2]
	local D = u._shape[1]
	task.desynchronize()
	local u_store = u._storage
	local v_store = v._storage
	local out_store = dot._storage
	for b = 1, B do
		local sum_uv, sum_uu, sum_vv = 0, 0, 0
		for i = 1, D do
			local idx = (i - 1) * B + b
			local val_u = u_store[idx]
			local val_v = v_store[idx]
			sum_uv += val_u * val_v
			sum_uu += val_u * val_u
			sum_vv += val_v * val_v
		end
		local norm_u = math.sqrt(sum_uu) + 1e-8
		local norm_v = math.sqrt(sum_vv) + 1e-8
		out_store[b] = sum_uv / (norm_u * norm_v)
	end
	task.synchronize()
	return dot
end

return function(cfg: {
	inputDim: number,
	actionDim: number,
	hiddenDim: number,
	goalDim: number?,
	horizon: number?,
	gammaW: number?,
	gammaM: number?,
	lr: number?,
	optimizerFactory: (params: { number }) -> Types.Optimizer,
})
	assert(cfg.inputDim, "Feudal: inputDim required")
	assert(cfg.actionDim, "Feudal: actionDim required")
	assert(cfg.hiddenDim, "Feudal: hiddenDim required")
	local inputDim = cfg.inputDim
	local actionDim = cfg.actionDim
	local hiddenDim = cfg.hiddenDim
	local goalDim = cfg.goalDim or hiddenDim
	local horizon = cfg.horizon or 10
	local _gammaW = cfg.gammaW or 0.99
	local _gammaM = cfg.gammaM or 0.99
	local _lr = cfg.lr or 1e-3
	local optimizerFactory = cfg.optimizerFactory
	assert(optimizerFactory, "Feudal: optimizerFactory required")
	local perceptor = Linear(inputDim, hiddenDim)
	local mgrLinear1 = Linear(hiddenDim, hiddenDim)
	local mgrLSTM = LSTM(hiddenDim, hiddenDim)
	local mgrGoalHead = Linear(hiddenDim, goalDim)
	local mgrValueHead = Linear(hiddenDim, 1)
	local wkrLinear1 = Linear(hiddenDim + goalDim, hiddenDim)
	local wkrLSTM = LSTM(hiddenDim, hiddenDim)
	local wkrActionHead = Linear(hiddenDim, actionDim)
	local wkrValueHead = Linear(hiddenDim, 1)
	local params = {}
	local function collectParams(mod)
		for _, p in ipairs(mod:parameters()) do
			table.insert(params, p)
		end
	end
	collectParams(perceptor)
	collectParams(mgrLinear1)
	collectParams(mgrLSTM)
	collectParams(mgrGoalHead)
	collectParams(mgrValueHead)
	collectParams(wkrLinear1)
	collectParams(wkrLSTM)
	collectParams(wkrActionHead)
	collectParams(wkrValueHead)
	local optimizer = optimizerFactory(params)
	local mgrH, mgrC
	local wkrH, wkrC
	local currentGoal
	local lastMgrUpdateStep = 0
	local stepCount = 0
	local startState
	local workerBuffer = {} -- {s, g, a, r_ext, r_int, ns, done}
	local managerBuffer = {} -- {s, g, r_sum, ns, done}
	local Agent = {}
	function Agent:reset(batchSize: number)
		batchSize = batchSize or 1
		mgrH = Tensor.zeros({ hiddenDim, batchSize }, "f64", true)
		mgrC = Tensor.zeros({ hiddenDim, batchSize }, "f64", true)
		wkrH = Tensor.zeros({ hiddenDim, batchSize }, "f64", true)
		wkrC = Tensor.zeros({ hiddenDim, batchSize }, "f64", true)
		currentGoal = Tensor.zeros({ goalDim, batchSize }, "f64", true)
		startState = nil
		stepCount = 0
		lastMgrUpdateStep = 0
		table.clear(workerBuffer)
		table.clear(managerBuffer)
	end
	function Agent:act(state: Types.Tensor)
		-- state: {inputDim, B}
		local B = state._shape[2]
		local z_raw = perceptor:forward(state)
		local z = Activations.ReLU(z_raw)
		if stepCount - lastMgrUpdateStep >= horizon or stepCount == 0 then
			local mgr_z = mgrLinear1:forward(z)
			mgr_z = Activations.Tanh(mgr_z)
			local h_new, c_new = mgrLSTM:forward(mgr_z, mgrH, mgrC)
			mgrH, mgrC = h_new, c_new
			local g = mgrGoalHead:forward(h_new)
			local g_norm = Tensor.zeros(g._shape, g._dtype, true)
			task.desynchronize()
			local g_store = g._storage
			local gn_store = g_norm._storage
			for b = 1, B do
				local sum_sq = 0
				for i = 1, goalDim do
					local val = g_store[(i - 1) * B + b]
					sum_sq += val * val
				end
				local scale = 1 / (math.sqrt(sum_sq) + 1e-8)
				for i = 1, goalDim do
					local idx = (i - 1) * B + b
					gn_store[idx] = g_store[idx] * scale
				end
			end
			task.synchronize()
			currentGoal = g_norm
			lastMgrUpdateStep = stepCount
			startState = state:detach()
		end
		-- z: {hiddenDim, B}, goal: {goalDim, B}
		local combinedInput = Tensor.zeros({ hiddenDim + goalDim, B }, state._dtype, true)
		task.desynchronize()
		local z_store = z._storage
		local g_store = currentGoal._storage
		local c_store = combinedInput._storage
		for b = 1, B do
			for i = 1, hiddenDim do
				c_store[(i - 1) * B + b] = z_store[(i - 1) * B + b]
			end
			for i = 1, goalDim do
				c_store[(hiddenDim + i - 1) * B + b] = g_store[(i - 1) * B + b]
			end
		end
		task.synchronize()
		local wz = wkrLinear1:forward(combinedInput)
		local wh_new, wc_new = wkrLSTM:forward(wz, wkrH, wkrC)
		wkrH, wkrC = wh_new, wc_new
		local logits = wkrActionHead:forward(wh_new)
		local P = Softmax.forward(logits)
		local action = 1
		task.desynchronize()
		local r = math.random()
		local cum = 0.0
		for i = 1, actionDim do
			cum += P._storage[i]
			if r <= cum then
				action = i
				break
			end
		end
		task.synchronize()
		stepCount += 1
		return action, currentGoal
	end

	function Agent:observe(transition: {
		state: Types.Tensor,
		action: number,
		reward: number,
		nextState: Types.Tensor,
		done: boolean,
	})
		local s = transition.state
		local a = transition.action
		local r = transition.reward
		local ns = transition.nextState
		local d = transition.done
		local z = Activations.ReLU(perceptor:forward(s))
		local z_next = Activations.ReLU(perceptor:forward(ns))
		local dz = Tensor.zeros(z._shape, z._dtype, false)
		task.desynchronize()
		local zs, zns, dzs = z._storage, z_next._storage, dz._storage
		for i = 1, #zs do
			dzs[i] = zns[i] - zs[i]
		end
		task.synchronize()
		local cosim = cosineSimilarity(dz, currentGoal)
		local r_int = cosim._storage[1]
		table.insert(workerBuffer, {
			s = s,
			g = currentGoal,
			a = a,
			r_ext = r,
			r_int = r_int,
			ns = ns,
			d = d,
		})
		if d or (stepCount - lastMgrUpdateStep == 0) then
			local r_mgr, workerBuffer_len = 0, #workerBuffer
			for i = workerBuffer_len - (stepCount - lastMgrUpdateStep), workerBuffer_len do
				if workerBuffer[i] then
					r_mgr += workerBuffer[i].r_ext
				end
			end
			table.insert(managerBuffer, {
				s = startState,
				g = currentGoal,
				r = r_mgr,
				ns = ns,
				d = d,
			})
		end
	end
	function Agent:trainStep()
		optimizer:zeroGrad()
		local _loss = Tensor.zeros({ 1 }, "f64", true)
		table.clear(workerBuffer)
		table.clear(managerBuffer)
		return { loss = 0 }
	end
	return Agent
end
