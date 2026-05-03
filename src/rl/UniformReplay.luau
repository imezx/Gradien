--!native
--!optimize 2
local Tensor = require("../Tensor")
local Buffer = require("../Util/Buffer")
local Util = require("../Util")
local assert = Util.Assert

local function packBatch(self, idxs: { number })
	local B = #idxs
	assert(B > 0, "UniformReplay: empty index list")
	assert(self.stateDim, "UniformReplay: buffer is empty")
	local sd = self.stateDim

	local S = Tensor.zeros({ sd, B }, false)
	local NS = Tensor.zeros({ sd, B }, false)
	local A = table.create(B, 1)
	local R = table.create(B, 0.0)
	local D = table.create(B, false)

	task.desynchronize()
	for j = 1, B do
		local k = idxs[j]
		A[j], R[j], D[j] = self.A[k], self.R[k], self.D[k]
		Buffer.copyDenseToColumn(self.S[k], S._storage, j, B, sd)
		Buffer.copyDenseToColumn(self.NS[k], NS._storage, j, B, sd)
	end
	task.synchronize()

	local w = table.create(B, 1.0)
	for j = 1, B do
		w[j] = 1.0
	end

	return { S = S, A = A, R = R, NS = NS, D = D, indices = idxs, weights = w }
end

local function push(self, t)
	assert(t.state and t.nextState, "UniformReplay: state and nextState required")
	local stateTensor = t.state
	local nextStateTensor = t.nextState
	assert(stateTensor._storage and nextStateTensor._storage, "UniformReplay: states must be Tensors")
	if not self.stateDim then
		self.stateDim = #stateTensor._storage
	end
	local i = self.head
	self.S[i] = Buffer.writeDenseArray(stateTensor._storage, self.S[i])
	self.A[i] = t.action
	self.R[i] = t.reward
	self.NS[i] = Buffer.writeDenseArray(nextStateTensor._storage, self.NS[i])
	self.D[i] = t.done and true or false

	self.head += 1
	if self.head > self.capacity then
		self.head = 1
	end
	if self.size_ < self.capacity then
		self.size_ += 1
	end
end

local function size(self): number
	return self.size_
end

local function clear(self)
	self.size_, self.head = 0, 1
	self.stateDim = nil
	table.clear(self.S)
	table.clear(self.A)
	table.clear(self.R)
	table.clear(self.NS)
	table.clear(self.D)
end

local function canSample(self, B: number): boolean
	return self.size_ >= B
end

local function sample(self, B: number)
	assert(self.size_ >= B, "UniformReplay: not enough samples")
	assert(self.stateDim, "UniformReplay: buffer is empty")
	local idxs = table.create(B, 0)
	task.desynchronize()
	for j = 1, B do
		idxs[j] = math.random(1, self.size_)
	end
	task.synchronize()
	return packBatch(self, idxs)
end

--@class UniformReplay
--@field capacity integer
--@field size_ integer
--@field head integer
--@field S {any}
--@field A {number}
--@field R {number}
--@field NS {any}
--@field D {boolean}
--@field stateDim number?
local function new(capacity: number)
	assert(capacity >= 1, "UniformReplay: capacity must be >= 1")
	return {
		capacity = capacity,
		size_ = 0,
		head = 1,
		S = {},
		A = {},
		R = {},
		NS = {},
		D = {},
		stateDim = nil,
		push = push,
		size = size,
		clear = clear,
		canSample = canSample,
		sample = sample,
	}
end

return function(capacity: number)
	return new(capacity)
end
