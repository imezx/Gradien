--!optimize 2
local Tensor = require("../Tensor")

--@class EMA
--@field decay number
--@field shadow {any}
--@field backup {any}?
local EMA = {}
EMA.__index = EMA

--@param params {any}
--@param decay number? @default 0.999
--@return EMA
function EMA.new(params, decay: number?)
	local self = setmetatable({}, EMA)
	self.decay = decay or 0.999
	self.shadow = {}
	for i = 1, #params do
		local t = params[i]
		local buf = Tensor.zeros(t._shape, t._dtype, false)
		for k = 1, #t._storage do buf._storage[k] = t._storage[k] end
		self.shadow[i] = buf
	end
	return self
end

function EMA:update(params)
	task.desynchronize()
	local m = self.decay
	for i = 1, #params do
		local t = params[i]
		local s = self.shadow[i]
		for k = 1, #t._storage do
			s._storage[k] = m * s._storage[k] + (1 - m) * t._storage[k]
		end
	end
	task.synchronize()
end

function EMA:apply(params)
	self.backup = {}
	for i = 1, #params do
		local t = params[i]
		local b = Tensor.zeros(t._shape, t._dtype, false)
		for k = 1, #t._storage do
			b._storage[k] = t._storage[k]
			t._storage[k] = self.shadow[i]._storage[k]
		end
		self.backup[i] = b
	end
end

function EMA:restore(params)
	if not self.backup then return end
	for i = 1, #params do
		local t = params[i]
		for k = 1, #t._storage do
			t._storage[k] = self.backup[i]._storage[k]
		end
	end
	self.backup = nil
end

return function(params, decay)
	return EMA.new(params, decay)
end
