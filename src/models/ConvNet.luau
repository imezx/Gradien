--!optimize 2
local Types = require("../Types")
local Utils = require("./Utils")
local Util = require("../Util")
local Conv2d = require("../nn/Conv2d")
local BatchNorm2d = require("../nn/BatchNorm2d")
local MaxPool2d = require("../nn/MaxPool2d")
local AvgPool2d = require("../nn/AvgPool2d")
local Dropout = require("../nn/Dropout")
local Flatten = require("../nn/Flatten")
local Linear = require("../nn/Linear")
local Sequential = require("../nn/Sequential")

local assert = Util.Assert

export type PoolConfig = {
	kind: "max" | "avg",
	size: number?,
	stride: number?,
}

export type ConvBlockConfig = {
	outChannels: number,
	kernelSize: number?,
	activation: Utils.ActivationName?,
	useBatchNorm: boolean?,
	pool: PoolConfig?,
	dropout: number?,
}

export type ConvNetConfig = {
	inChannels: number,
	inputHeight: number,
	inputWidth: number,
	numClasses: number,
	blocks: { ConvBlockConfig }?,
	headDims: { number }?,
	activation: Utils.ActivationName?,
	dropout: number?,
	headDropout: number?,
	finalActivation: Utils.ActivationName?,
	weightInit: ((number, number) -> number)?,
}

local function defaultBlocks(): { ConvBlockConfig }
	return {
		{ outChannels = 32, kernelSize = 3, pool = { kind = "max", size = 2 } },
		{ outChannels = 64, kernelSize = 3, pool = { kind = "max", size = 2 } },
		{ outChannels = 128, kernelSize = 3, pool = { kind = "max", size = 2 } },
	}
end

local function applyConvSize(size: number, kernel: number): number
	local nextSize = size - kernel + 1
	assert(nextSize > 0, "ConvNet: spatial dimension collapsed (conv)")
	return nextSize
end

local function applyPoolSize(size: number, cfg: PoolConfig): number
	local k = cfg.size or 2
	local stride = cfg.stride or k
	local nextSize = math.floor((size - k) / stride) + 1
	assert(nextSize > 0, "ConvNet: spatial dimension collapsed (pool)")
	return nextSize
end

return function(config: ConvNetConfig): Types.Module<Types.Tensor, Types.Tensor>
	assert(config.inChannels and config.inputHeight and config.inputWidth, "ConvNet: input spec required")
	assert(config.numClasses, "ConvNet: numClasses required")
	local blocks = if config.blocks and #config.blocks > 0 then config.blocks else defaultBlocks()

	local H, W = config.inputHeight, config.inputWidth
	local C = config.inChannels
	for _, block: ConvBlockConfig in ipairs(blocks) do
		local kernel = block.kernelSize or 3
		H = applyConvSize(H, kernel)
		W = applyConvSize(W, kernel)
		if block.pool then
			H = applyPoolSize(H, block.pool)
			W = applyPoolSize(W, block.pool)
		end
		C = block.outChannels
	end
	local flattenDim = C * H * W
	assert(flattenDim > 0, "ConvNet: invalid flatten dimension")

	local convLayers = {}
	local trainables: { any } = {}
	local inChannels = config.inChannels
	for _, block: ConvBlockConfig in ipairs(blocks) do
		local kernel = block.kernelSize or 3
		local conv = Conv2d(inChannels, block.outChannels, kernel, kernel)
		table.insert(convLayers, conv)
		table.insert(trainables, conv)

		if block.useBatchNorm ~= false then
			local bn = BatchNorm2d(block.outChannels)
			table.insert(convLayers, bn)
			table.insert(trainables, bn)
		end

		local blockActivation: Utils.ActivationName?
		if block.activation ~= nil then
			blockActivation = block.activation
		elseif config.activation ~= nil then
			blockActivation = config.activation
		else
			blockActivation = nil
		end
		table.insert(convLayers, Utils.makeActivationModule(blockActivation))

		if block.pool then
			local poolCfg = block.pool
			local size = poolCfg.size or 2
			local stride = poolCfg.stride or size
			local poolModule = if poolCfg.kind == "avg"
				then AvgPool2d(size, size, stride)
				else MaxPool2d(size, size, stride)
			table.insert(convLayers, poolModule)
			table.insert(trainables, poolModule)
		end

		local dropProb = block.dropout or config.dropout or 0
		if dropProb > 0 then
			local drop = Dropout(math.clamp(dropProb, 0, 0.95))
			table.insert(convLayers, drop)
			table.insert(trainables, drop)
		end

		inChannels = block.outChannels
	end

	local featureExtractor = Sequential(convLayers)
	local flatten = Flatten()
	table.insert(trainables, flatten)

	local headLayers = {}
	local headDims = config.headDims or { 256 }
	local headDrop = math.clamp(config.headDropout or config.dropout or 0, 0, 0.95)
	local lastDim = flattenDim
	for _, hidden in ipairs(headDims) do
		assert(hidden > 0, "ConvNet: head hidden dims must be positive")
		local lin = Linear(lastDim, hidden, config.weightInit)
		table.insert(headLayers, lin)
		table.insert(trainables, lin)
		table.insert(headLayers, Utils.makeActivationModule(config.activation))
		if headDrop > 0 then
			local drop = Dropout(headDrop)
			table.insert(headLayers, drop)
			table.insert(trainables, drop)
		end
		lastDim = hidden
	end
	local classifier = Linear(lastDim, config.numClasses, config.weightInit)
	table.insert(headLayers, classifier)
	table.insert(trainables, classifier)
	if config.finalActivation then
		table.insert(headLayers, Utils.makeActivationModule(config.finalActivation))
	end

	local head = Sequential(headLayers)

	return {
		forward = function(_, x: Types.Tensor): Types.Tensor
			local h = featureExtractor:forward(x)
			h = flatten:forward(h)
			return head:forward(h)
		end,
		parameters = function(_): { Types.Tensor }
			local params = {}
			Utils.collectParams(params, featureExtractor)
			Utils.collectParams(params, head)
			return params
		end,
		train = function(_, mode: boolean?)
			Utils.trainAll(trainables, mode)
		end,
	}
end
