--!native
--!optimize 2
local Tensor = require("../Tensor")

local function update(self, params)
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

local function apply(self, params)
	self.backup = {}
	for i = 1, #params do
		local t = params[i]
		local b = Tensor.zeros(t._shape, false)
		for k = 1, #t._storage do
			b._storage[k] = t._storage[k]
			t._storage[k] = self.shadow[i]._storage[k]
		end
		self.backup[i] = b
	end
end

local function restore(self, params)
	if not self.backup then
		return
	end
	for i = 1, #params do
		local t = params[i]
		for k = 1, #t._storage do
			t._storage[k] = self.backup[i]._storage[k]
		end
	end
	self.backup = nil
end

local function new(params, decay: number?)
	local shadow = {}
	for i = 1, #params do
		local t = params[i]
		local buf = Tensor.zeros(t._shape, false)
		for k = 1, #t._storage do
			buf._storage[k] = t._storage[k]
		end
		shadow[i] = buf
	end
	return {
		decay = decay or 0.999,
		shadow = shadow,
		backup = nil,
		update = update,
		apply = apply,
		restore = restore,
		setLr = function() end,
	}
end

return function(params, decay)
	return new(params, decay)
end
