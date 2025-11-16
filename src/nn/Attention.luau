--!optimize 2
local Tensor = require("../Tensor")
local Tape = require("../autograd/Tape")
local Types = require("../Types")
local Linear = require("./Linear")
local SoftmaxOps = require("../ops/Softmax")

type TensorT = Types.Tensor
type Module = Types.Module<TensorT, TensorT>
type Random = typeof(Random.new())

export type AttentionOptions = {
	qkvInit: ((number, number) -> number)?,
	projInit: ((number, number) -> number)?,
	attentionDropout: number?, -- dropout prob applied to attention weights during training
	dropoutRNG: Random?, -- optional RNG source for dropout sampling
	causal: boolean?, -- enable built-in causal masking
}
local NEG_INF = -1e9

local Attention = {}
Attention.__index = Attention

local function scaleTensor(x: TensorT, factor: number): TensorT
	local y = Tensor.zeros(x._shape, x._dtype, x._requiresGrad)
	task.desynchronize()
	for i = 1, #x._storage do
		y._storage[i] = x._storage[i] * factor
	end
	task.synchronize()
	if not x._requiresGrad then
		return y
	end
	return Tape.makeNode(y, { x }, function(gy: TensorT)
		if not x._requiresGrad then
			return
		end
		x._grad = x._grad or Tensor.zeros(x._shape)
		local grad = x._grad :: TensorT
		task.desynchronize()
		for i = 1, #gy._storage do
			grad._storage[i] += gy._storage[i] * factor
		end
		task.synchronize()
	end)
end

local function addTensors(a: TensorT, b: TensorT): TensorT
	assert(#a._shape == #b._shape, "addTensors: shape rank mismatch")
	for i = 1, #a._shape do
		assert(a._shape[i] == b._shape[i], "addTensors: shape mismatch")
	end
	local requiresGrad = (a._requiresGrad or b._requiresGrad) == true
	local out = Tensor.zeros(a._shape, a._dtype, requiresGrad)
	task.desynchronize()
	for i = 1, #a._storage do
		out._storage[i] = a._storage[i] + b._storage[i]
	end
	task.synchronize()
	if not requiresGrad then
		return out
	end
	return Tape.makeNode(out, { a, b }, function(gy: TensorT)
		task.desynchronize()
		if a._requiresGrad then
			a._grad = a._grad or Tensor.zeros(a._shape)
			local gradA = a._grad :: TensorT
			for i = 1, #gy._storage do
				gradA._storage[i] += gy._storage[i]
			end
		end
		if b._requiresGrad then
			b._grad = b._grad or Tensor.zeros(b._shape)
			local gradB = b._grad :: TensorT
			for i = 1, #gy._storage do
				gradB._storage[i] += gy._storage[i]
			end
		end
		task.synchronize()
	end)
end

local function dropoutTensor(x: TensorT, dropProb: number, rng: Random?): TensorT
	if dropProb <= 0 then
		return x
	end
	local keep = 1 - dropProb
	assert(keep > 0, "dropoutTensor: keep probability must be > 0")
	local mask = table.create(#x._storage)
	local y = Tensor.zeros(x._shape, x._dtype, x._requiresGrad)
	task.desynchronize()
	for i = 1, #x._storage do
		local keepSample = ((rng or Random.new()):NextNumber(0, 1) < keep) and 1 or 0
		mask[i] = keepSample
		y._storage[i] = x._storage[i] * keepSample / keep
	end
	task.synchronize()
	if not x._requiresGrad then
		return y
	end
	return Tape.makeNode(y, { x }, function(gy: TensorT)
		if not x._requiresGrad then
			return
		end
		x._grad = x._grad or Tensor.zeros(x._shape)
		local grad = x._grad :: TensorT
		task.desynchronize()
		for i = 1, #gy._storage do
			grad._storage[i] += gy._storage[i] * (mask[i] / keep)
		end
		task.synchronize()
	end)
end

local function softmaxRows(x: TensorT): TensorT
	local rows, cols = x._shape[1], x._shape[2]
	local y = SoftmaxOps.forward(x)
	if not x._requiresGrad then
		return y
	end
	return Tape.makeNode(y, { x }, function(gy: TensorT)
		if not x._requiresGrad then
			return
		end
		x._grad = x._grad or Tensor.zeros(x._shape)
		local grad = x._grad :: TensorT
		task.desynchronize()
		local yStore = y._storage
		local gyStore = gy._storage
		local gxStore = grad._storage
		for i = 1, rows do
			local offset = (i - 1) * cols
			local dot = 0
			for j = 1, cols do
				local idx = offset + j
				dot += yStore[idx] * gyStore[idx]
			end
			for j = 1, cols do
				local idx = offset + j
				gxStore[idx] += yStore[idx] * (gyStore[idx] - dot)
			end
		end
		task.synchronize()
	end)
end

local function concatHeads(heads: { TensorT }, headDim: number, tokens: number): TensorT
	assert(#heads > 0, "concatHeads: expected at least one head")
	local embedDim = headDim * #heads
	local requiresGrad = false
	for _, h in heads do
		if h._requiresGrad then
			requiresGrad = true
			break
		end
	end
	local y = Tensor.zeros({ embedDim, tokens }, heads[1]._dtype, requiresGrad)
	task.desynchronize()
	local yStore = y._storage
	for idxHead = 1, #heads do
		local h = heads[idxHead]
		assert(h._shape[1] == headDim, "concatHeads: head dim mismatch")
		assert(h._shape[2] == tokens, "concatHeads: token dim mismatch")
		local hStore = h._storage
		local rowOffset = (idxHead - 1) * headDim
		for i = 1, headDim do
			for j = 1, tokens do
				local dstRow = rowOffset + i
				local dstIdx = (dstRow - 1) * tokens + j
				local srcIdx = (i - 1) * tokens + j
				yStore[dstIdx] = hStore[srcIdx]
			end
		end
	end
	task.synchronize()
	if not requiresGrad then
		return y
	end
	return Tape.makeNode(y, heads, function(gy: TensorT)
		task.desynchronize()
		for idxHead = 1, #heads do
			local h = heads[idxHead]
			if h._requiresGrad then
				h._grad = h._grad or Tensor.zeros(h._shape)
				local hg = (h._grad :: TensorT)._storage
				local rowOffset = (idxHead - 1) * headDim
				for i = 1, headDim do
					for j = 1, tokens do
						local dstRow = rowOffset + i
						local gyIdx = (dstRow - 1) * tokens + j
						local srcIdx = (i - 1) * tokens + j
						hg[srcIdx] += gy._storage[gyIdx]
					end
				end
			end
		end
		task.synchronize()
	end)
end

type DropoutConfig = {
	prob: number,
	rng: Random?,
	enabled: boolean?,
}

local function scaledDotProduct(
	q: TensorT,
	k: TensorT,
	v: TensorT,
	mask: TensorT?,
	scaleOverride: number?,
	dropCfg: DropoutConfig?
): (TensorT, TensorT)
	local headDim = q._shape[1]
	local queryLen = q._shape[2]
	local keyLen = k._shape[2]
	assert(k._shape[1] == headDim, "scaledDotProduct: Q/K dim mismatch")
	assert(v._shape[1] == headDim, "scaledDotProduct: Q/V dim mismatch")
	assert(v._shape[2] == keyLen, "scaledDotProduct: K/V length mismatch")
	local scale = scaleOverride or (1 / math.sqrt(headDim))
	local scores = Tape.matmul(Tensor.transpose(q), k) -- {queryLen, keyLen}
	scores = scaleTensor(scores, scale)
	if mask then
		assert(mask._shape[1] == queryLen and mask._shape[2] == keyLen, "scaledDotProduct: mask shape mismatch")
		scores = addTensors(scores, mask)
	end
	local attn = softmaxRows(scores)
	if dropCfg and dropCfg.prob > 0 and dropCfg.enabled ~= false then
		attn = dropoutTensor(attn, dropCfg.prob, dropCfg.rng)
	end
	local context = Tensor.transpose(Tape.matmul(attn, Tensor.transpose(v)))
	return context, attn
end

function Attention.new(embedDim: number, numHeads: number, opts: AttentionOptions?): Module
	assert(numHeads >= 1, "Attention: numHeads must be >= 1")
	assert(embedDim % numHeads == 0, "Attention: embedDim must be divisible by numHeads")
	opts = opts or {}
	local options = opts :: AttentionOptions
	local headDim = math.floor(embedDim / numHeads)
	local self = setmetatable({
		_embedDim = embedDim,
		_numHeads = numHeads,
		_headDim = headDim,
		_scale = 1 / math.sqrt(headDim),
		_qkv = Linear(embedDim, embedDim * 3, options.qkvInit),
		_proj = Linear(embedDim, embedDim, options.projInit),
		_attentionDropout = math.clamp(options.attentionDropout or 0, 0, 0.999),
		_dropoutRNG = options.dropoutRNG or Random.new(),
		_train = true,
		_causal = options.causal == true,
		_causalMaskCache = {} :: { [Types.DType]: { [number]: TensorT } },
		_lastAttn = nil,
	}, Attention)
	return self
end

function Attention:forward(x: TensorT, mask: TensorT?): TensorT
	local tokens = x._shape[2]
	local embedDim = self._embedDim
	local qkv = self._qkv:forward(x) -- {3*D, tokens}
	assert(qkv._shape[1] == embedDim * 3, "Attention: unexpected projection shape")
	local q = Tensor.narrow(qkv, 1, 1, embedDim)
	local k = Tensor.narrow(qkv, 1, embedDim + 1, embedDim)
	local v = Tensor.narrow(qkv, 1, 2 * embedDim + 1, embedDim)
	local effMask = mask
	if not effMask and self._causal then
		effMask = self:_getCausalMask(tokens, q._dtype)
	end
	local dropCfg: DropoutConfig?
	if self._attentionDropout > 0 and self._train then
		dropCfg = {
			prob = self._attentionDropout,
			rng = self._dropoutRNG,
			enabled = true,
		}
	end
	local headOutputs = table.create(self._numHeads)
	local attnWeights = table.create(self._numHeads)
	for h = 0, self._numHeads - 1 do
		local offset = h * self._headDim + 1
		local qh = Tensor.narrow(q, 1, offset, self._headDim)
		local kh = Tensor.narrow(k, 1, offset, self._headDim)
		local vh = Tensor.narrow(v, 1, offset, self._headDim)
		local ctx, weights = scaledDotProduct(qh, kh, vh, effMask, self._scale, dropCfg)
		headOutputs[h + 1] = ctx
		attnWeights[h + 1] = weights
	end
	self._lastAttn = attnWeights
	local merged = concatHeads(headOutputs, self._headDim, tokens)
	return self._proj:forward(merged)
end

function Attention:parameters(): { TensorT }
	local params = {}
	for _, t in self._qkv:parameters() do
		table.insert(params, t)
	end
	for _, t in self._proj:parameters() do
		table.insert(params, t)
	end
	return params
end

function Attention:getLastAttention(): { TensorT }?
	return self._lastAttn
end

function Attention:train(mode: boolean?)
	self._train = (mode ~= false)
end

function Attention:setCausal(enabled: boolean?)
	self._causal = enabled ~= false
end

function Attention:_getCausalMask(tokens: number, dtype: Types.DType?): TensorT
	dtype = dtype or "f64"
	local cacheByDtype = self._causalMaskCache[dtype]
	if not cacheByDtype then
		cacheByDtype = {}
		self._causalMaskCache[dtype] = cacheByDtype
	end
	local mask = cacheByDtype[tokens]
	if mask then
		return mask
	end
	local newMask = Tensor.zeros({ tokens, tokens }, dtype, false)
	task.desynchronize()
	local store = newMask._storage
	for i = 1, tokens do
		local offset = (i - 1) * tokens
		for j = i + 1, tokens do
			store[offset + j] = NEG_INF
		end
	end
	task.synchronize()
	cacheByDtype[tokens] = newMask
	return newMask
end

local exported = {
	new = function(embedDim: number, numHeads: number, opts: AttentionOptions?): Module
		return Attention.new(embedDim, numHeads, opts)
	end,
	scaledDotProduct = scaledDotProduct,
}

setmetatable(exported, {
	__call = function(_, embedDim: number, numHeads: number, opts: AttentionOptions?): Module
		return Attention.new(embedDim, numHeads, opts)
	end,
})

return exported
