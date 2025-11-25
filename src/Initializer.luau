--!native
--!optimize 2
local Types = require("./Types")
local Initializer = {}

local function _randn(): number
	local u1 = math.random()
	local u2 = math.random()
	if u1 <= 1e-7 then
		u1 = 1e-7
	end
	return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
end

function Initializer.xavierUniform(W: Types.Tensor)
	task.desynchronize()
	local fanIn = W._shape[2]
	local fanOut = W._shape[1]
	local bound = math.sqrt(6 / (fanIn + fanOut))
	for i = 1, #W._storage do
		W._storage[i] = (math.random() * 2 - 1) * bound
	end
	task.synchronize()
end

function Initializer.kaimingUniform(W: Types.Tensor)
	task.desynchronize()
	local fanIn = W._shape[2]
	local bound = math.sqrt(6 / fanIn)
	for i = 1, #W._storage do
		W._storage[i] = (math.random() * 2 - 1) * bound
	end
	task.synchronize()
end

function Initializer.heUniform(W: Types.Tensor)
	Initializer.kaimingUniform(W)
end

-- N(0, 2 / fanIn)
function Initializer.heNormal(W: Types.Tensor)
	task.desynchronize()
	local fanIn = W._shape[2]
	local std = math.sqrt(2 / fanIn)
	for i = 1, #W._storage do
		W._storage[i] = _randn() * std
	end
	task.synchronize()
end

-- U(-sqrt(3 / fanIn), sqrt(3 / fanIn))
function Initializer.lecunUniform(W: Types.Tensor)
	task.desynchronize()
	local fanIn = W._shape[2]
	local bound = math.sqrt(3 / fanIn)
	for i = 1, #W._storage do
		W._storage[i] = (math.random() * 2 - 1) * bound
	end
	task.synchronize()
end

-- N(0, 1 / fanIn)
function Initializer.lecunNormal(W: Types.Tensor)
	task.desynchronize()
	local fanIn = W._shape[2]
	local std = math.sqrt(1 / fanIn)
	for i = 1, #W._storage do
		W._storage[i] = _randn() * std
	end
	task.synchronize()
end

function Initializer.zeros(B: Types.Tensor)
	task.desynchronize()
	for i = 1, #B._storage do
		B._storage[i] = 0
	end
	task.synchronize()
end

return Initializer
