--!native
--!optimize 2
local Tensor = require("../Tensor")
local Tape = require("../autograd/Tape")
local Types = require("../Types")
local Activations = require("../nn/Activations")
local Util = require("../Util")

local assert = Util.Assert

export type ActivationName = "relu" | "gelu" | "tanh" | "silu" | "sigmoid" | "mish" | "elu" | "leaky_relu" | "selu"

type Module = Types.Module<Types.Tensor, Types.Tensor>

local Utils = {}

local ActivationLookup: { [string]: (Types.Tensor) -> Types.Tensor } = {
	relu = Activations.ReLU,
	gelu = Activations.GELU,
	tanh = Activations.Tanh,
	silu = Activations.SiLU,
	sigmoid = Activations.Sigmoid,
	mish = Activations.Mish,
	elu = function(x: Types.Tensor): Types.Tensor
		return Activations.ELU(x)
	end,
	leaky_relu = function(x: Types.Tensor): Types.Tensor
		return Activations.LeakyReLU(x, 0.01)
	end,
	selu = Activations.SeLU,
}

local function normalizeKey(name: string?): string
	if not name then
		return "relu"
	end
	return string.lower(name)
end

function Utils.getActivationFn(name: ActivationName?): (Types.Tensor) -> Types.Tensor
	local fn = ActivationLookup[normalizeKey(name)]
	assert(fn, `Unsupported activation '{name}'.`)
	return fn
end

function Utils.makeActivationModule(name: ActivationName?): Module
	local fn = Utils.getActivationFn(name)
	return {
		forward = function(_, x: Types.Tensor): Types.Tensor
			return fn(x)
		end,
		parameters = function(_): { Types.Tensor }
			return {}
		end,
		train = function(_) end,
	}
end

function Utils.collectParams(dst: { Types.Tensor }, module: Module?)
	if not module then
		return
	end
	local ok, params = pcall(function()
		return module:parameters()
	end)
	if ok and params then
		for _, p in ipairs(params) do
			table.insert(dst, p)
		end
	end
end

function Utils.collectFromList(dst: { Types.Tensor }, modules: { Module? })
	for _, m in ipairs(modules) do
		Utils.collectParams(dst, m)
	end
end

function Utils.trainAll(modules: { any }, mode: boolean?)
	for _, module in ipairs(modules) do
		if type(module) == "table" then
			local trainFn = (module :: any).train
			if type(trainFn) == "function" then
				trainFn(module, mode)
			end
		end
	end
end

function Utils.addTensors(a: Types.Tensor, b: Types.Tensor): Types.Tensor
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
	return Tape.makeNode(out, { a, b }, function(gy: Types.Tensor)
		task.desynchronize()
		if a._requiresGrad then
			a._grad = a._grad or Tensor.zeros(a._shape)
			for i = 1, #gy._storage do
				(a._grad :: Types.Tensor)._storage[i] += gy._storage[i]
			end
		end
		if b._requiresGrad then
			b._grad = b._grad or Tensor.zeros(b._shape)
			for i = 1, #gy._storage do
				(b._grad :: Types.Tensor)._storage[i] += gy._storage[i]
			end
		end
		task.synchronize()
	end)
end

return Utils
