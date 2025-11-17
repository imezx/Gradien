--!optimize 2
local Types = require("../Types")
local Utils = require("./Utils")
local Util = require("../Util")
local Linear = require("../nn/Linear")
local LayerNorm = require("../nn/LayerNorm")
local Dropout = require("../nn/Dropout")
local Sequential = require("../nn/Sequential")

local assert = Util.Assert

export type MLPConfig = {
	inputDim: number,
	outputDim: number,
	hiddenDims: { number }?,
	activation: Utils.ActivationName?,
	finalActivation: Utils.ActivationName?,
	dropout: number?,
	layerNorm: boolean?,
	weightInit: ((number, number) -> number)?,
}

return function(config: MLPConfig): Types.Module<Types.Tensor, Types.Tensor>
	assert(config.inputDim, "MLP: inputDim is required")
	assert(config.outputDim, "MLP: outputDim is required")

	local hidden = if config.hiddenDims and #config.hiddenDims > 0
		then table.clone(config.hiddenDims)
		else {
			math.max(config.inputDim, config.outputDim),
		}
	local dropProb = math.clamp(config.dropout or 0, 0, 0.95)

	local layers = {}
	local trainables: { any } = {}
	local inDim = config.inputDim

	for _, hiddenDim in ipairs(hidden) do
		assert(hiddenDim > 0, "MLP: hidden dimensions must be positive")
		local linear = Linear(inDim, hiddenDim, config.weightInit)
		table.insert(layers, linear)
		table.insert(trainables, linear)

		if config.layerNorm then
			local norm = LayerNorm(hiddenDim)
			table.insert(layers, norm)
			table.insert(trainables, norm)
		end

		table.insert(layers, Utils.makeActivationModule(config.activation))

		if dropProb > 0 then
			local drop = Dropout(dropProb)
			table.insert(layers, drop)
			table.insert(trainables, drop)
		end

		inDim = hiddenDim
	end

	local head = Linear(inDim, config.outputDim, config.weightInit)
	table.insert(layers, head)
	table.insert(trainables, head)

	if config.finalActivation then
		table.insert(layers, Utils.makeActivationModule(config.finalActivation))
	end

	local seq = Sequential(layers)

	return {
		forward = function(_, x: Types.Tensor): Types.Tensor
			return seq:forward(x)
		end,
		parameters = function(_): { Types.Tensor }
			return seq:parameters()
		end,
		train = function(_, mode: boolean?)
			Utils.trainAll(trainables, mode)
		end,
	}
end
