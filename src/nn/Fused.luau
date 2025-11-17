--!optimize 2
local Tensor = require("../Tensor")
local Types = require("../Types")
local Linear = require("./Linear")
local Activations = require("./Activations")
local Dropout = require("./Dropout")
local Util = require("../Util")
local assert = Util.Assert

type Tensor = Types.Tensor
type Module = Types.Module

local Fused = {}

--@param x Tensor {inDim, batch}
--@param W Tensor {outDim, inDim}
--@param b Tensor {outDim, 1}
--@returns Tensor {outDim, batch} = ReLU(Wx + b)
function Fused.linearReLU(x: Tensor, W: Tensor, b: Tensor): Tensor
	local inDim = x._shape[1]
	local batch = x._shape[2]
	local outDim = W._shape[1]

	local y = Tensor.zeros({ outDim, batch }, x._dtype, false)
	local yStore = y._storage
	local xStore = x._storage
	local wStore = W._storage
	local bStore = b._storage

	task.desynchronize()
	for i = 1, outDim do
		local bias = bStore[i]
		for j = 1, batch do
			local sum = 0.0
			-- dot(W[i,:], x[:,j])
			for k = 1, inDim do
				local wIdx = (i - 1) * inDim + k -- W row-major
				local xIdx = (k - 1) * batch + j -- x {rows=inDim, batch}
				sum += wStore[wIdx] * xStore[xIdx]
			end

			sum += bias
			if sum < 0 then
				sum = 0
			end

			local yIdx = (i - 1) * batch + j
			yStore[yIdx] = sum
		end
	end
	task.synchronize()

	return y
end

local function gelu_scalar(x: number): number
	-- 0.5 * x * (1 + tanh(√(2/π)*(x + 0.044715 x^3)))
	local c = math.sqrt(2 / math.pi)
	local x3 = x * x * x
	local inner = c * (x + 0.044715 * x3)
	local t = math.tanh(inner)
	return 0.5 * x * (1 + t)
end

--@param x Tensor {inDim, batch}
--@param W Tensor {outDim, inDim}
--@param b Tensor {outDim, 1}
--@returns Tensor {outDim, batch} = GELU(Wx + b)
function Fused.linearGELU(x: Tensor, W: Tensor, b: Tensor): Tensor
	local inDim = x._shape[1]
	local batch = x._shape[2]
	local outDim = W._shape[1]

	local y = Tensor.zeros({ outDim, batch }, x._dtype, false)
	local yStore = y._storage
	local xStore = x._storage
	local wStore = W._storage
	local bStore = b._storage

	task.desynchronize()
	for i = 1, outDim do
		local bias = bStore[i]
		for j = 1, batch do
			local sum = 0.0
			for k = 1, inDim do
				local wIdx = (i - 1) * inDim + k
				local xIdx = (k - 1) * batch + j
				sum += wStore[wIdx] * xStore[xIdx]
			end

			sum += bias
			sum = gelu_scalar(sum)

			local yIdx = (i - 1) * batch + j
			yStore[yIdx] = sum
		end
	end
	task.synchronize()

	return y
end

--@param x Tensor {inDim, batch}
--@param W Tensor {outDim, inDim}
--@param b Tensor {outDim, 1}
--@param p number dropout probability in [0,1)
--@param rng (() -> number)? (optional) → number in [0,1), default math.random
--@returns { Tensor, Tensor }
function Fused.linearDropoutReLU(x: Tensor, W: Tensor, b: Tensor, p: number, rng: (() -> number)?): (Tensor, Tensor)
	rng = rng or math.random

	local keep = 1.0 - p
	assert(keep > 0, "Fused.linearDropoutReLU: keep prob must be > 0")

	local inDim = x._shape[1]
	local batch = x._shape[2]
	local outDim = W._shape[1]

	local y = Tensor.zeros({ outDim, batch }, x._dtype, false)
	local mask = Tensor.zeros({ outDim, batch }, x._dtype, false)

	local yStore = y._storage
	local mStore = mask._storage
	local xStore = x._storage
	local wStore = W._storage
	local bStore = b._storage

	local scale = 1.0 / keep

	task.desynchronize()
	for i = 1, outDim do
		local bias = bStore[i]
		for j = 1, batch do
			local sum = 0.0
			for k = 1, inDim do
				local wIdx = (i - 1) * inDim + k
				local xIdx = (k - 1) * batch + j
				sum += wStore[wIdx] * xStore[xIdx]
			end

			sum += bias

			local idx = (i - 1) * batch + j
			local r = rng()
			if r < keep and sum > 0 then
				mStore[idx] = 1.0
				yStore[idx] = sum * scale
			else
				mStore[idx] = 0.0
				yStore[idx] = 0.0
			end
		end
	end
	task.synchronize()

	return y, mask
end

local Modules = {}
Modules.__index = Modules

-- Linear -> ReLU as one block

local LinearReLU = {}
LinearReLU.__index = LinearReLU

export type LinearReLU = {
	_linear: Module,
}

function LinearReLU.new(inDim: number, outDim: number, init: ((number, number) -> number)?): Module
	local self = setmetatable({}, LinearReLU)
	self._linear = Linear(inDim, outDim, init)

	return (self :: any) :: Module
end

function LinearReLU:parameters()
	return self._linear:parameters()
end

function LinearReLU:forward(x: Tensor): Tensor
	local z = self._linear:forward(x)
	return Activations.ReLU(z)
end

Modules.LinearReLU = LinearReLU

--Linear -> GELU as one block

local LinearGELU = {}
LinearGELU.__index = LinearGELU

export type LinearGELU = {
	_linear: Module,
}

function LinearGELU.new(inDim: number, outDim: number, init: ((number, number) -> number)?): Module
	local self = setmetatable({}, LinearGELU)
	self._linear = Linear(inDim, outDim, init)
	return (self :: any) :: Module
end

function LinearGELU:parameters()
	return self._linear:parameters()
end

function LinearGELU:forward(x: Tensor): Tensor
	local z = self._linear:forward(x)
	return Activations.GELU(z)
end

Modules.LinearGELU = LinearGELU

local LinearDropoutReLU = {}
LinearDropoutReLU.__index = LinearDropoutReLU

export type LinearDropoutReLU = {
	_linear: Module,
	_drop: Module,
}

--Linear -> Dropout -> ReLU
--@param outDim number
--@param p number
--@param init ((number, number) -> number)?
--@return Module
function LinearDropoutReLU.new(inDim: number, outDim: number, p: number, init: ((number, number) -> number)?): Module
	local self = setmetatable({}, LinearDropoutReLU)
	self._linear = Linear(inDim, outDim, init)
	self._drop = Dropout(p)
	return (self :: any) :: Module
end

function LinearDropoutReLU:parameters()
	local params = self._linear:parameters()
	return params
end

function LinearDropoutReLU:forward(x: Tensor): Tensor
	local z = self._linear:forward(x)
	local d = self._drop:forward(z)
	return Activations.ReLU(d)
end

Modules.LinearDropoutReLU = LinearDropoutReLU
Fused.Modules = Modules

return Fused
