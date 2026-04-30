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

-- U(-sqrt(6 / (fanIn + fanOut)), sqrt(6 / (fanIn + fanOut)))
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

-- N(0, sqrt(2 / (fanIn + fanOut)))
function Initializer.xavierNormal(W: Types.Tensor)
	task.desynchronize()
	local fanIn = W._shape[2]
	local fanOut = W._shape[1]
	local std = math.sqrt(2 / (fanIn + fanOut))
	for i = 1, #W._storage do
		W._storage[i] = _randn() * std
	end
	task.synchronize()
end

-- U(-sqrt(6 / fanIn), sqrt(6 / fanIn))
function Initializer.kaimingUniform(W: Types.Tensor)
	task.desynchronize()
	local fanIn = W._shape[2]
	local bound = math.sqrt(6 / fanIn)
	for i = 1, #W._storage do
		W._storage[i] = (math.random() * 2 - 1) * bound
	end
	task.synchronize()
end

Initializer.heUniform = Initializer.kaimingUniform

-- N(0, sqrt(2 / fanIn))
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

-- N(0, sqrt(1 / fanIn))
function Initializer.lecunNormal(W: Types.Tensor)
	task.desynchronize()
	local fanIn = W._shape[2]
	local std = math.sqrt(1 / fanIn)
	for i = 1, #W._storage do
		W._storage[i] = _randn() * std
	end
	task.synchronize()
end

function Initializer.orthogonal(W: Types.Tensor, gain: number?)
	local g = gain or 1.0
	task.desynchronize()
	local rows = W._shape[1]
	local cols = W._shape[2]
	-- build a rows x cols random normal matrix as column vectors
	local vecs: { { number } } = {}
	for j = 1, cols do
		local v = table.create(rows, 0.0)
		for i = 1, rows do
			v[i] = _randn()
		end
		vecs[j] = v
	end
	-- gram-schmidt
	for j = 1, cols do
		local v = vecs[j]
		for k = 1, j - 1 do
			local u = vecs[k]
			local dot = 0.0
			for i = 1, rows do
				dot += u[i] * v[i]
			end
			for i = 1, rows do
				v[i] -= dot * u[i]
			end
		end
		local norm2 = 0.0
		for i = 1, rows do
			norm2 += v[i] * v[i]
		end
		local inv = g / math.sqrt(math.max(norm2, 1e-12))
		for i = 1, rows do
			v[i] *= inv
		end
		vecs[j] = v
	end
	-- write column-major into row-major storage {rows, cols}
	for j = 1, cols do
		for i = 1, rows do
			W._storage[(i - 1) * cols + j] = vecs[j][i]
		end
	end
	task.synchronize()
end

-- fill with a constant value.
function Initializer.constant(W: Types.Tensor, value: number)
	task.desynchronize()
	for i = 1, #W._storage do
		W._storage[i] = value
	end
	task.synchronize()
end

-- uniform U(a, b).
function Initializer.uniform(W: Types.Tensor, a: number, b: number)
	task.desynchronize()
	local range = b - a
	for i = 1, #W._storage do
		W._storage[i] = a + math.random() * range
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
