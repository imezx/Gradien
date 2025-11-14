--!optimize 2
local Tensor = require("../Tensor")
local Tape = require("../autograd/Tape")
local Types = require("../Types")
local Linear = require("./Linear")

type TensorT = Types.Tensor
type Module = Types.Module<TensorT, TensorT>

export type AttentionOptions = {
	qkvInit: ((number, number) -> number)?,
	projInit: ((number, number) -> number)?,
}

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

local function softmaxRows(x: TensorT): TensorT
	local rows, cols = x._shape[1], x._shape[2]
	local y = Tensor.zeros(x._shape, x._dtype, x._requiresGrad)
	task.desynchronize()
	for i = 1, rows do
		local offset = (i - 1) * cols
		local m = -1 / 0
		for j = 1, cols do
			local idx = offset + j
			local v = x._storage[idx]
			if v > m then
				m = v
			end
		end
		local sum = 0
		for j = 1, cols do
			local idx = offset + j
			local e = math.exp(x._storage[idx] - m)
			y._storage[idx] = e
			sum += e
		end
		local inv = 1 / math.max(sum, 1e-12)
		for j = 1, cols do
			local idx = offset + j
			y._storage[idx] *= inv
		end
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
	assert(next(heads), "concatHeads: expected at least one head")
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

local function scaledDotProduct(
	q: TensorT,
	k: TensorT,
	v: TensorT,
	mask: TensorT?,
	scaleOverride: number?
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
	local headOutputs = table.create(self._numHeads)
	local attnWeights = table.create(self._numHeads)
	for h = 0, self._numHeads - 1 do
		local offset = h * self._headDim + 1
		local qh = Tensor.narrow(q, 1, offset, self._headDim)
		local kh = Tensor.narrow(k, 1, offset, self._headDim)
		local vh = Tensor.narrow(v, 1, offset, self._headDim)
		local ctx, weights = scaledDotProduct(qh, kh, vh, mask, self._scale)
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
