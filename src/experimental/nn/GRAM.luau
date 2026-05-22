--!native
--!optimize 2
-- arXiv:2605.19376v2 [cs.AI] 20 May 2026
-- Generative Recursive Reasoning
local Tensor = require("../../Tensor")
local Tape = require("../../autograd/Tape")
local Types = require("../../Types")
local Linear = require("../../nn/Linear")
local LayerNorm = require("../../nn/LayerNorm")
local GatedMLP = require("../../nn/GatedMLP")
local Attention = require("../../nn/Attention")
local Activations = require("../../nn/Activations")
local Utils = require("../../models/Utils")
local Initializer = require("../../Initializer")

export type Config = {
	embedDim: number,
	vocabSize: number,
	numHeads: number?,
	mlpDim: number?,
	K: number?, -- low-level steps (default 4)
	T: number?, -- high-level steps (default 3)
	beta: number?, -- KL coefficient (default 0.05)
	alpha: number?, -- KL balancing weight (default 0.8)
	useAttention: boolean?, -- default true
}

-- input: A shape [da, L], B shape [db, L] -> output: shape [da + db, L]
local function concat(A: Types.Tensor, B: Types.Tensor): Types.Tensor
	local da = A._shape[1]
	local db = B._shape[1]
	local L = A._shape[2]
	assert(B._shape[2] == L, "concat: token dimension mismatch")
	local requiresGrad = A._requiresGrad or B._requiresGrad
	local out = Tensor.zeros({ da + db, L }, requiresGrad)
	task.desynchronize()
	local as = A._storage
	local bs = B._storage
	local os = out._storage
	-- copy A
	for i = 1, da do
		local a_offset = (i - 1) * L
		local o_offset = (i - 1) * L
		for j = 1, L do
			os[o_offset + j] = as[a_offset + j]
		end
	end
	-- copy B
	for i = 1, db do
		local b_offset = (i - 1) * L
		local o_offset = (da + i - 1) * L
		for j = 1, L do
			os[o_offset + j] = bs[b_offset + j]
		end
	end
	task.synchronize()
	if not requiresGrad then
		return out
	end
	return Tape.makeNode(out, { A, B }, function(gy)
		local gys = gy._storage
		if A._requiresGrad then
			if not A._grad then
				A._grad = Tensor.zeros(A._shape)
			end
			local gas = (A._grad :: any)._storage
			task.desynchronize()
			for i = 1, da do
				local a_offset = (i - 1) * L
				local o_offset = (i - 1) * L
				for j = 1, L do
					gas[a_offset + j] += gys[o_offset + j]
				end
			end
			task.synchronize()
		end
		if B._requiresGrad then
			if not B._grad then
				B._grad = Tensor.zeros(B._shape)
			end
			local gbs = (B._grad :: any)._storage
			task.desynchronize()
			for i = 1, db do
				local b_offset = (i - 1) * L
				local o_offset = (da + i - 1) * L
				for j = 1, L do
					gbs[b_offset + j] += gys[o_offset + j]
				end
			end
			task.synchronize()
		end
	end)
end

local function ensureGrad(t: Types.Tensor): Types.Tensor
	if not t._grad then
		t._grad = Tensor.zeros(t._shape, false)
	end
	return t._grad :: any
end

-- samples epsilon ~ N(mu, exp(2*log_sigma))
local function reparameterize(mu: Types.Tensor, log_sigma: Types.Tensor): Types.Tensor
	local shape = mu._shape
	local eta = Tensor.randn(shape, false)
	local requiresGrad = mu._requiresGrad or log_sigma._requiresGrad
	local out = Tensor.zeros(shape, requiresGrad)
	task.desynchronize()
	local ms = mu._storage
	local lss = log_sigma._storage
	local es = eta._storage
	local os = out._storage
	for i = 1, #ms do
		local sigma_val = math.exp(lss[i])
		os[i] = ms[i] + sigma_val * es[i]
	end
	task.synchronize()
	if not requiresGrad then
		return out
	end
	return Tape.makeNode(out, { mu, log_sigma }, function(gy)
		local gys = gy._storage
		if mu._requiresGrad then
			local gmu = ensureGrad(mu)
			local gmus = gmu._storage
			task.desynchronize()
			for i = 1, #gys do
				gmus[i] += gys[i]
			end
			task.synchronize()
		end
		if log_sigma._requiresGrad then
			local gls = ensureGrad(log_sigma)
			local glss = gls._storage
			task.desynchronize()
			for i = 1, #gys do
				local sigma_val = math.exp(lss[i])
				glss[i] += gys[i] * sigma_val * es[i]
			end
			task.synchronize()
		end
	end)
end

local function klDivergence(
	mu_q: Types.Tensor,
	log_sigma_q: Types.Tensor,
	mu_p: Types.Tensor,
	log_sigma_p: Types.Tensor
): Types.Tensor
	local shape = mu_q._shape
	local D = shape[1]
	local L = shape[2]
	local size = D * L
	local kl_val = 0
	task.desynchronize()
	local mq = mu_q._storage
	local lsq = log_sigma_q._storage
	local mp = mu_p._storage
	local lsp = log_sigma_p._storage
	for i = 1, size do
		local sq2 = math.exp(2 * lsq[i])
		local sp2 = math.exp(2 * lsp[i])
		local diff = mq[i] - mp[i]
		kl_val += 0.5 * (2 * lsp[i] - 2 * lsq[i] + (sq2 + diff * diff) / sp2 - 1)
	end
	task.synchronize()
	local out = Tensor.fromArray(
		{ kl_val },
		{ 1 },
		mu_q._requiresGrad or log_sigma_q._requiresGrad or mu_p._requiresGrad or log_sigma_p._requiresGrad
	)
	if not out._requiresGrad then
		return out
	end
	return Tape.makeNode(out, { mu_q, log_sigma_q, mu_p, log_sigma_p }, function(gy)
		local g = gy._storage[1]
		if mu_q._requiresGrad then
			local gmu = ensureGrad(mu_q)
			local gmus = gmu._storage
			task.desynchronize()
			for i = 1, size do
				local sp2 = math.exp(2 * lsp[i])
				gmus[i] += g * (mq[i] - mp[i]) / sp2
			end
			task.synchronize()
		end
		if log_sigma_q._requiresGrad then
			local gls = ensureGrad(log_sigma_q)
			local glss = gls._storage
			task.desynchronize()
			for i = 1, size do
				local sq2 = math.exp(2 * lsq[i])
				local sp2 = math.exp(2 * lsp[i])
				glss[i] += g * (sq2 / sp2 - 1)
			end
			task.synchronize()
		end
		if mu_p._requiresGrad then
			local gmu = ensureGrad(mu_p)
			local gmus = gmu._storage
			task.desynchronize()
			for i = 1, size do
				local sp2 = math.exp(2 * lsp[i])
				gmus[i] -= g * (mq[i] - mp[i]) / sp2
			end
			task.synchronize()
		end
		if log_sigma_p._requiresGrad then
			local gls = ensureGrad(log_sigma_p)
			local glss = gls._storage
			task.desynchronize()
			for i = 1, size do
				local sq2 = math.exp(2 * lsq[i])
				local sp2 = math.exp(2 * lsp[i])
				local diff = mq[i] - mp[i]
				glss[i] += g * (1 - (sq2 + diff * diff) / sp2)
			end
			task.synchronize()
		end
	end)
end

local function add_kl_to_logits(logits: Types.Tensor, kl_loss: Types.Tensor): Types.Tensor
	local out = Tensor.zeros(logits._shape, logits._requiresGrad or kl_loss._requiresGrad)
	task.desynchronize()
	for i = 1, #logits._storage do
		out._storage[i] = logits._storage[i]
	end
	task.synchronize()
	if not out._requiresGrad then
		return out
	end
	return Tape.makeNode(out, { logits, kl_loss }, function(gy)
		if logits._requiresGrad then
			local g_logits = ensureGrad(logits)
			task.desynchronize()
			for i = 1, #gy._storage do
				g_logits._storage[i] += gy._storage[i]
			end
			task.synchronize()
		end
		if kl_loss._requiresGrad then
			local g_kl = ensureGrad(kl_loss)
			task.desynchronize()
			g_kl._storage[1] += 1.0
			task.synchronize()
		end
	end)
end

local function buildTransformerBlock(embedDim: number, numHeads: number, mlpDim: number, useAttention: boolean)
	local norm1 = LayerNorm(embedDim)
	local attn = if useAttention then Attention(embedDim, numHeads) else nil
	local norm2 = LayerNorm(embedDim)
	local ff1 = GatedMLP(embedDim, mlpDim, embedDim)
	local ff2 = GatedMLP(embedDim, mlpDim, embedDim)

	local function initLinear(l)
		Initializer.xavierUniform(l.W)
		Initializer.zeros(l.b)
	end

	if attn then
		initLinear(attn._qkv)
		initLinear(attn._proj)
	end
	initLinear(ff1.fc_in)
	initLinear(ff1.fc_out)
	initLinear(ff2.fc_in)
	initLinear(ff2.fc_out)

	local trainables = { norm1, norm2, ff1, ff2 }
	if attn then
		table.insert(trainables, attn)
	end

	local function forward(self, x: Types.Tensor): Types.Tensor
		local h = x
		if self.attn then
			local y = self.norm1:forward(h)
			y = self.attn:forward(y)
			h = Utils.addTensors(h, y)
		else
			local y = self.norm1:forward(h)
			y = self.ff1:forward(y)
			h = Utils.addTensors(h, y)
		end

		local z = self.norm2:forward(h)
		z = self.ff2:forward(z)
		h = Utils.addTensors(h, z)
		return h
	end

	local function parameters(self)
		local params = {}
		Utils.collectFromList(params, self.trainables)
		return params
	end

	local function train(self, mode: boolean?)
		Utils.trainAll(self.trainables, mode)
	end

	return {
		norm1 = norm1,
		attn = attn,
		ff1 = ff1,
		norm2 = norm2,
		ff2 = ff2,
		forward = forward,
		parameters = parameters,
		train = train,
		trainables = trainables,
	}
end

local function forward(self, x: Types.Tensor, y: Types.Tensor?): Types.Tensor
	local L = x._shape[2]
	local D = self._embedDim

	-- constant latent initialization z_0 = (h_0, l_0)
	if not self._h_0 or self._h_0._shape[2] ~= L then
		self._h_0 = Tensor.randn({ D, L }, false)
		self._l_0 = Tensor.randn({ D, L }, false)
	end

	local h_prev = self._h_0
	local l_prev = self._l_0

	-- accumulators
	local kl_terms = {}
	local act_steps = self._T
	local halted = false

	for t = 1, self._T do
		if halted then
			act_steps = t - 1
			break
		end

		-- inner loop
		local l_k = l_prev
		local context = Utils.addTensors(h_prev, x) -- h_{t-1} + e_x
		for k = 1, self._K do
			l_k = self.f_L:forward(Utils.addTensors(l_k, context))
		end
		local l_t = l_k

		-- outer loop
		local u_t = self.f_H:forward(Utils.addTensors(h_prev, l_t))

		-- noise parameters
		local norm_u = self.prior_norm:forward(u_t)
		local p1 = self.prior_fc1:forward(norm_u)
		local p2 = Activations.SwiGLUSplit(p1)
		local mu_p = self.prior_fc_mu:forward(p2)
		local log_sigma_p = self.prior_fc_logvar:forward(p2)

		local h_t
		if y and self._train then
			local norm_concat = self.post_norm:forward(concat(u_t, y))
			local post1 = self.post_fc1:forward(norm_concat)
			local post2 = Activations.SwiGLUSplit(post1)
			local mu_q = self.post_fc_mu:forward(post2)
			local log_sigma_q = self.post_fc_logvar:forward(post2)
			-- sampling
			local epsilon = reparameterize(mu_q, log_sigma_q)
			h_t = Utils.addTensors(u_t, epsilon)

			-- kl_q_fixed = KL(q || detach(p))
			-- kl_p_fixed = KL(detach(q) || p)
			local kl_q_fixed = klDivergence(mu_q, log_sigma_q, Tensor.detach(mu_p), Tensor.detach(log_sigma_p))
			local kl_p_fixed = klDivergence(Tensor.detach(mu_q), Tensor.detach(log_sigma_q), mu_p, log_sigma_p)

			local term1 = Utils.addTensors(kl_p_fixed, Tensor.zeros({ 1 }))
			local term2 = Utils.addTensors(kl_q_fixed, Tensor.zeros({ 1 }))

			local scaled_term1 = Tensor.zeros({ 1 }, term1._requiresGrad)
			task.desynchronize()
			scaled_term1._storage[1] = term1._storage[1] * self._alpha
			task.synchronize()
			if term1._requiresGrad then
				scaled_term1 = Tape.makeNode(scaled_term1, { term1 }, function(gy)
					if term1._requiresGrad then
						ensureGrad(term1)._storage[1] += gy._storage[1] * self._alpha
					end
				end)
			end

			local scaled_term2 = Tensor.zeros({ 1 }, term2._requiresGrad)
			task.desynchronize()
			scaled_term2._storage[1] = term2._storage[1] * (1 - self._alpha)
			task.synchronize()
			if term2._requiresGrad then
				scaled_term2 = Tape.makeNode(scaled_term2, { term2 }, function(gy)
					if term2._requiresGrad then
						ensureGrad(term2)._storage[1] += gy._storage[1] * (1 - self._alpha)
					end
				end)
			end

			local kl_balanced = Utils.addTensors(scaled_term1, scaled_term2)

			local kl_scaled = Tensor.zeros({ 1 }, kl_balanced._requiresGrad)
			task.desynchronize()
			kl_scaled._storage[1] = kl_balanced._storage[1] * self._beta
			task.synchronize()
			if kl_balanced._requiresGrad then
				kl_scaled = Tape.makeNode(kl_scaled, { kl_balanced }, function(gy)
					if kl_balanced._requiresGrad then
						ensureGrad(kl_balanced)._storage[1] += gy._storage[1] * self._beta
					end
				end)
			end

			table.insert(kl_terms, kl_scaled)
		else
			local epsilon = reparameterize(mu_p, log_sigma_p)
			h_t = Utils.addTensors(u_t, epsilon)

			local act_logits = self.act_head:forward(h_t)
			local act_sum = 0
			task.desynchronize()
			for i = 1, L do
				act_sum += act_logits._storage[i]
			end
			task.synchronize()
			local mean_act = act_sum / L
			local halt_prob = 1 / (1 + math.exp(-mean_act))
			if halt_prob > 0.5 then
				halted = true
				h_prev = h_t
				l_prev = l_t
				act_steps = t
				break
			end
		end

		h_prev = h_t
		l_prev = l_t
	end

	self._last_act_steps = act_steps

	local logits = self.lm_head:forward(h_prev)

	local lprm_logits = self.lprm_head:forward(h_prev)
	local lprm_sum = 0
	task.desynchronize()
	for i = 1, L do
		lprm_sum += lprm_logits._storage[i]
	end
	task.synchronize()
	self._last_lprm_value = lprm_sum / L

	if #kl_terms > 0 then
		local total_kl = kl_terms[1]
		for i = 2, #kl_terms do
			total_kl = Utils.addTensors(total_kl, kl_terms[i])
		end
		logits = add_kl_to_logits(logits, total_kl)
	end

	return logits
end

local function parameters(self): { Types.Tensor }
	local params = {}
	Utils.collectParams(params, self.f_L)
	Utils.collectParams(params, self.f_H)
	Utils.collectParams(params, self.prior_norm)
	Utils.collectParams(params, self.prior_fc1)
	Utils.collectParams(params, self.prior_fc_mu)
	Utils.collectParams(params, self.prior_fc_logvar)
	Utils.collectParams(params, self.post_norm)
	Utils.collectParams(params, self.post_fc1)
	Utils.collectParams(params, self.post_fc_mu)
	Utils.collectParams(params, self.post_fc_logvar)
	Utils.collectParams(params, self.act_head)
	Utils.collectParams(params, self.lprm_head)
	Utils.collectParams(params, self.lm_head)
	return params
end

local function train(self, mode: boolean?)
	self._train = (mode ~= false)
	Utils.trainAll({
		self.f_L,
		self.f_H,
		self.prior_norm,
		self.prior_fc1,
		self.prior_fc_mu,
		self.prior_fc_logvar,
		self.post_norm,
		self.post_fc1,
		self.post_fc_mu,
		self.post_fc_logvar,
		self.act_head,
		self.lprm_head,
		self.lm_head,
	}, mode)
end

local function initLinear(l)
	Initializer.xavierUniform(l.W)
	Initializer.zeros(l.b)
end

local function initLinearZeros(l)
	Initializer.zeros(l.W)
	Initializer.zeros(l.b)
end

local function generate(self, x: Types.Tensor, N: number): (Types.Tensor, number)
	assert(type(N) == "number" and N >= 1, "GRAM:generate - N must be a number greater than or equal to 1")
	local old_train = self._train
	self:train(false) -- set to eval/prior mode

	local best_logits = nil
	local best_score = -math.huge

	for i = 1, N do
		Tape.noGrad(function()
			local logits = self:forward(x)
			local score = self._last_lprm_value
			if i == 1 or (score > best_score) or (best_score ~= best_score and score == score) then
				best_score = score
				best_logits = logits
			end
		end)
	end

	self:train(old_train)
	return best_logits :: Types.Tensor, best_score
end

return {
	new = function(cfg: Config)
		assert(cfg.embedDim, "GRAM: embedDim required")
		assert(cfg.vocabSize, "GRAM: vocabSize required")

		local embedDim = cfg.embedDim
		local vocabSize = cfg.vocabSize
		local numHeads = cfg.numHeads or 8
		local mlpDim = cfg.mlpDim or embedDim
		local K = cfg.K or 4
		local T = cfg.T or 3
		local beta = cfg.beta or 0.05
		local alpha = cfg.alpha or 0.8
		local useAttention = cfg.useAttention ~= false

		local f_L = buildTransformerBlock(embedDim, numHeads, mlpDim, useAttention)
		local f_H = buildTransformerBlock(embedDim, numHeads, mlpDim, useAttention)

		local prior_norm = LayerNorm(embedDim)
		local post_norm = LayerNorm(2 * embedDim)

		-- MLPs: u_t -> mu_p, logvar_p
		local prior_fc1 = Linear(embedDim, 2 * embedDim)
		local prior_fc_mu = Linear(embedDim, embedDim)
		local prior_fc_logvar = Linear(embedDim, embedDim)

		-- MLPs: [u_t, e_y] -> mu_q, logvar_q
		local post_fc1 = Linear(2 * embedDim, 2 * embedDim)
		local post_fc_mu = Linear(embedDim, embedDim)
		local post_fc_logvar = Linear(embedDim, embedDim)

		-- Heads
		local act_head = Linear(embedDim, 1) -- ACT halting logit
		local lprm_head = Linear(embedDim, 1) -- LPRM score
		local lm_head = Linear(embedDim, vocabSize) -- Logits output

		initLinear(prior_fc1)
		initLinear(prior_fc_mu)
		initLinearZeros(prior_fc_logvar)
		initLinear(post_fc1)
		initLinear(post_fc_mu)
		initLinearZeros(post_fc_logvar)
		initLinear(act_head)
		initLinear(lprm_head)
		initLinear(lm_head)

		return {
			_embedDim = embedDim,
			_vocabSize = vocabSize,
			_K = K,
			_T = T,
			_beta = beta,
			_alpha = alpha,
			_useAttention = useAttention,

			f_L = f_L,
			f_H = f_H,
			prior_norm = prior_norm,
			prior_fc1 = prior_fc1,
			prior_fc_mu = prior_fc_mu,
			prior_fc_logvar = prior_fc_logvar,
			post_norm = post_norm,
			post_fc1 = post_fc1,
			post_fc_mu = post_fc_mu,
			post_fc_logvar = post_fc_logvar,
			act_head = act_head,
			lprm_head = lprm_head,
			lm_head = lm_head,

			_train = true,
			_h_0 = nil,
			_l_0 = nil,
			_last_act_steps = 0,
			_last_lprm_value = 0,

			forward = forward,
			parameters = parameters,
			train = train,
			generate = generate,
		}
	end,
}
