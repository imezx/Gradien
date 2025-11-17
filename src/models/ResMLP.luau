--!optimize 2
local Types = require("../Types")
local Utils = require("./Utils")
local Util = require("../Util")
local Linear = require("../nn/Linear")
local LayerNorm = require("../nn/LayerNorm")
local Dropout = require("../nn/Dropout")
local Sequential = require("../nn/Sequential")

local assert = Util.Assert

export type ResMLPConfig = {
	inputDim: number,
	outputDim: number,
	embedDim: number?,
	hiddenDim: number?,
	numBlocks: number?,
	activation: Utils.ActivationName?,
	dropout: number?,
	layerNorm: boolean?,
	finalLayerNorm: boolean?,
	weightInit: ((number, number) -> number)?,
}

type Block = {
	norm: Types.Module<Types.Tensor, Types.Tensor>?,
	ff: Types.Module<Types.Tensor, Types.Tensor>,
	trainables: { any },
}

local function buildBlock(
	embedDim: number,
	hiddenDim: number,
	activation: Utils.ActivationName?,
	dropProb: number,
	useLayerNorm: boolean?,
	init: ((number, number) -> number)?
): Block
	local norm = if useLayerNorm then LayerNorm(embedDim) else nil
	local layers = {}
	local trainables: { any } = {}

	local function insertModule(m)
		table.insert(layers, m)
		table.insert(trainables, m)
	end

	insertModule(Linear(embedDim, hiddenDim, init))
	insertModule(Utils.makeActivationModule(activation))
	if dropProb > 0 then
		insertModule(Dropout(dropProb))
	end
	insertModule(Linear(hiddenDim, embedDim, init))
	if dropProb > 0 then
		insertModule(Dropout(dropProb))
	end

	local ff = Sequential(layers)
	if norm then
		table.insert(trainables, norm)
	end

	return {
		norm = norm,
		ff = ff,
		trainables = trainables,
	}
end

local function buildModel(config: ResMLPConfig)
	assert(config.inputDim and config.outputDim, "ResMLP: inputDim and outputDim required")
	local embedDim = config.embedDim or config.inputDim
	local hiddenDim = config.hiddenDim or (embedDim * 4)
	local blocks = math.max(1, config.numBlocks or 4)
	local dropout = math.clamp(config.dropout or 0, 0, 0.95)

	local inputProj = if config.inputDim ~= embedDim then Linear(config.inputDim, embedDim, config.weightInit) else nil
	local outputHead = Linear(embedDim, config.outputDim, config.weightInit)
	local finalNorm = if config.finalLayerNorm then LayerNorm(embedDim) else nil

	local stack = table.create(blocks)
	for i = 1, blocks do
		stack[i] = buildBlock(embedDim, hiddenDim, config.activation, dropout, config.layerNorm, config.weightInit)
	end

	local trainables: { any } = {}
	if inputProj then
		table.insert(trainables, inputProj)
	end
	for _, block in ipairs(stack) do
		for _, m in ipairs(block.trainables) do
			table.insert(trainables, m)
		end
	end
	if finalNorm then
		table.insert(trainables, finalNorm)
	end
	table.insert(trainables, outputHead)

	local function forward(_, x: Types.Tensor): Types.Tensor
		local h = x
		if inputProj then
			h = inputProj:forward(h)
		end
		for _, block in ipairs(stack) do
			local y = h
			if block.norm then
				y = block.norm:forward(y)
			end
			y = block.ff:forward(y)
			h = Utils.addTensors(h, y)
		end
		if finalNorm then
			h = finalNorm:forward(h)
		end
		return outputHead:forward(h)
	end

	local function parameters(_): { Types.Tensor }
		local params = {}
		if inputProj then
			Utils.collectParams(params, inputProj)
		end
		for _, block in ipairs(stack) do
			if block.norm then
				Utils.collectParams(params, block.norm)
			end
			Utils.collectParams(params, block.ff)
		end
		if finalNorm then
			Utils.collectParams(params, finalNorm)
		end
		Utils.collectParams(params, outputHead)
		return params
	end

	local function train(_, mode: boolean?)
		Utils.trainAll(trainables, mode)
	end

	local module = {
		forward = forward,
		parameters = parameters,
		train = train,
	}
	return module
end

return setmetatable({
	new = function(config: ResMLPConfig)
		return buildModel(config)
	end,
}, {
	__call = function(_, config: ResMLPConfig)
		return buildModel(config)
	end,
})
