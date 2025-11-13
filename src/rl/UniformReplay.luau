--!optimize 2
local Tensor = require("../Tensor")

--@class UniformReplay
--@field capacity integer
--@field size_ integer
--@field head integer
--@field S {any}
--@field A {number}
--@field R {number}
--@field NS {any}
--@field D {boolean}
local UR = {}
UR.__index = UR

--@param capacity integer
--@return UniformReplay
function UR.new(capacity: number)
	assert(capacity >= 1, "UniformReplay: capacity must be >= 1")
	local self = setmetatable({}, UR)
	self.capacity = capacity
	self.size_ = 0
	self.head = 1
	self.S, self.A, self.R, self.NS, self.D = {}, {}, {}, {}, {}
	return self
end

--@param t {state:any, action:number, reward:number, nextState:any, done:boolean?}
function UR:push(t)
	local i = self.head
	self.S[i] = t.state
	self.A[i] = t.action
	self.R[i] = t.reward
	self.NS[i] = t.nextState
	self.D[i] = t.done and true or false

	self.head += 1
	if self.head > self.capacity then self.head = 1 end
	if self.size_ < self.capacity then self.size_ += 1 end
end

function UR:size(): number
	return self.size_
end

function UR:clear()
	self.size_, self.head = 0, 1
	table.clear(self.S); table.clear(self.A); table.clear(self.R); table.clear(self.NS); table.clear(self.D)
end

function UR:canSample(B: number): boolean
	return self.size_ >= B
end

local function packBatch(self, idxs: {number})
	local B = #idxs
	assert(B > 0, "UniformReplay: empty index list")
	local sd = self.S[idxs[1]].__shape and self.S[idxs[1]].__shape[1] or self.S[idxs[1]]._shape[1]

	local S = Tensor.zeros({ sd, B })
	local NS = Tensor.zeros({ sd, B })
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
			S._storage[(r - 1) * B + j] = s._storage[r]
			NS._storage[(r - 1) * B + j] = ns._storage[r]
		end
	end
	task.synchronize()

	-- weights = 1 (no importance correction needed for uniform)
	local w = table.create(B, 1.0)
	for j = 1, B do w[j] = 1.0 end

	return { S = S, A = A, R = R, NS = NS, D = D, indices = idxs, weights = w }
end

--@param B integer
--@return {S:any,A:{number},R:{number},NS:any,D:{boolean},indices:{number},weights:{number}}
function UR:sample(B: number)
	assert(self.size_ >= B, "UniformReplay: not enough samples")
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
