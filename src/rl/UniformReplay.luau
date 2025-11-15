--!optimize 2
local Tensor = require("../Tensor")

---@class UniformReplay
---@field capacity integer
---@field size_ integer
---@field head integer
---@field S {any}
---@field A {number}
---@field R {number}
---@field NS {any}
---@field D {boolean}
---@field stateDim number?
---@field stateDType string?
local UR = {}
UR.__index = UR

local function ensureBuffer(buf: { number }?, dim: number)
	if not buf or #buf ~= dim then
		return table.create(dim, 0)
	end
	return buf
end

local function copyStorage(src: { number }, dst: { number }, dim: number)
	table.move(src, 1, dim, 1, dst)
	return
end

---@param capacity integer
---@return UniformReplay
function UR.new(capacity: number)
	assert(capacity >= 1, "UniformReplay: capacity must be >= 1")
	local self = setmetatable({}, UR)
	self.capacity = capacity
	self.size_ = 0
	self.head = 1
	self.S, self.A, self.R, self.NS, self.D = {}, {}, {}, {}, {}
	self.stateDim = nil
	self.stateDType = nil
	return self
end

---@param t {state:any, action:number, reward:number, nextState:any, done:boolean?}
function UR:push(t)
	assert(t.state and t.nextState, "UniformReplay: state and nextState required")
	local stateTensor = t.state
	local nextStateTensor = t.nextState
	assert(stateTensor._storage and nextStateTensor._storage, "UniformReplay: states must be Tensors")
	if not self.stateDim then
		self.stateDim = #stateTensor._storage
		self.stateDType = stateTensor._dtype
	end
	local sd = self.stateDim
	local i = self.head
	local stateBuf = ensureBuffer(self.S[i], sd)
	local nextStateBuf = ensureBuffer(self.NS[i], sd)
	task.desynchronize()
	copyStorage(stateTensor._storage, stateBuf, sd)
	copyStorage(nextStateTensor._storage, nextStateBuf, sd)
	task.synchronize()
	self.S[i] = stateBuf
	self.A[i] = t.action
	self.R[i] = t.reward
	self.NS[i] = nextStateBuf
	self.D[i] = t.done and true or false

	self.head += 1
	if self.head > self.capacity then
		self.head = 1
	end
	if self.size_ < self.capacity then
		self.size_ += 1
	end
end

function UR:size(): number
	return self.size_
end

function UR:clear()
	self.size_, self.head = 0, 1
	self.stateDim, self.stateDType = nil, nil
	table.clear(self.S)
	table.clear(self.A)
	table.clear(self.R)
	table.clear(self.NS)
	table.clear(self.D)
end

function UR:canSample(B: number): boolean
	return self.size_ >= B
end

local function packBatch(self, idxs: { number })
	local B = #idxs
	assert(B > 0, "UniformReplay: empty index list")
	assert(self.stateDim, "UniformReplay: buffer is empty")
	local sd = self.stateDim
	local dtype = self.stateDType

	local S = Tensor.zeros({ sd, B }, dtype, false)
	local NS = Tensor.zeros({ sd, B }, dtype, false)
	local A = table.create(B, 1)
	local R = table.create(B, 0.0)
	local D = table.create(B, false)

	task.desynchronize()
	for j = 1, B do
		local k = idxs[j]
		local s = self.S[k]
		local ns = self.NS[k]
		A[j], R[j], D[j] = self.A[k], self.R[k], self.D[k]
		for r = 1, sd do
			S._storage[(r - 1) * B + j] = s[r]
			NS._storage[(r - 1) * B + j] = ns[r]
		end
	end
	task.synchronize()

	-- weights = 1 (no importance correction needed for uniform)
	local w = table.create(B, 1.0)
	for j = 1, B do
		w[j] = 1.0
	end

	return { S = S, A = A, R = R, NS = NS, D = D, indices = idxs, weights = w }
end

---@param B integer
---@return {S:any,A:{number},R:{number},NS:any,D:{boolean},indices:{number},weights:{number}}
function UR:sample(B: number)
	assert(self.size_ >= B, "UniformReplay: not enough samples")
	assert(self.stateDim, "UniformReplay: buffer is empty")
	local idxs = table.create(B, 0)
	-- uniform with replacement (fast, low alloc)
	task.desynchronize()
	for j = 1, B do
		idxs[j] = math.random(1, self.size_)
	end
	task.synchronize()
	return packBatch(self, idxs)
end

return function(capacity: number)
	return UR.new(capacity)
end
