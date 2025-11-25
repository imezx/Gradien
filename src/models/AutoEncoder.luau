--!native
--!optimize 2
local Types = require("../Types")
local Utils = require("./Utils")
local Util = require("../Util")
local Linear = require("../nn/Linear")
local LayerNorm = require("../nn/LayerNorm")
local Dropout = require("../nn/Dropout")
local Sequential = require("../nn/Sequential")

local assert = Util.Assert

export type AutoEncoderConfig = {
	inputDim: number,
	latentDim: number,
	hiddenDims: { number }?,
	activation: Utils.ActivationName?,
	dropout: number?,
	layerNorm: boolean?,
	finalActivation: Utils.ActivationName?,
	weightInit: ((number, number) -> number)?,
}

local AutoEncoder = {}
AutoEncoder.__index = AutoEncoder

local function buildStack(
	inDim: number,
	hiddenDims: { number },
	outDim: number,
	activation: Utils.ActivationName?,
	dropProb: number,
	useLayerNorm: boolean?,
	init: ((number, number) -> number)?
): (Types.Module<Types.Tensor, Types.Tensor>, { any })
	local layers = {}
	local trainables: { any } = {}
	local function insert(m)
		table.insert(layers, m)
		table.insert(trainables, m)
	end
	local current = inDim
	for _, hidden in ipairs(hiddenDims) do
		insert(Linear(current, hidden, init))
		if useLayerNorm then
			insert(LayerNorm(hidden))
		end
		insert(Utils.makeActivationModule(activation))
		if dropProb > 0 then
			insert(Dropout(dropProb))
		end
		current = hidden
	end
	insert(Linear(current, outDim, init))
	return Sequential(layers), trainables
end

function AutoEncoder.new(cfg: AutoEncoderConfig): Types.Module<Types.Tensor, Types.Tensor>
	assert(cfg.inputDim and cfg.latentDim, "AutoEncoder: inputDim and latentDim required")
	local hidden = if cfg.hiddenDims and #cfg.hiddenDims > 0
		then table.clone(cfg.hiddenDims)
		else {
			math.max(cfg.inputDim, cfg.latentDim * 2),
		}
	local revHidden = table.clone(hidden)
	for i = 1, math.floor(#revHidden / 2) do
		local j = #revHidden - i + 1
		revHidden[i], revHidden[j] = revHidden[j], revHidden[i]
	end
	local drop = math.clamp(cfg.dropout or 0, 0, 0.95)

	local encoder, encTrain =
		buildStack(cfg.inputDim, hidden, cfg.latentDim, cfg.activation, drop, cfg.layerNorm, cfg.weightInit)
	local decoder, decTrain =
		buildStack(cfg.latentDim, revHidden, cfg.inputDim, cfg.activation, drop, cfg.layerNorm, cfg.weightInit)
	local finalAct = if cfg.finalActivation then Utils.makeActivationModule(cfg.finalActivation) else nil

	local trainables: { any } = {}
	for _, m in ipairs(encTrain) do
		table.insert(trainables, m)
	end
	for _, m in ipairs(decTrain) do
		table.insert(trainables, m)
	end
	if finalAct then
		table.insert(trainables, finalAct)
	end

	local self = setmetatable({
		_encoder = encoder,
		_decoder = decoder,
		_finalAct = finalAct,
		_trainables = trainables,
	}, AutoEncoder)

	return self
end

function AutoEncoder:encode(x: Types.Tensor): Types.Tensor
	return self._encoder:forward(x)
end

function AutoEncoder:decode(z: Types.Tensor): Types.Tensor
	local out = self._decoder:forward(z)
	if self._finalAct then
		out = self._finalAct:forward(out)
	end
	return out
end

function AutoEncoder:forward(x: Types.Tensor): Types.Tensor
	local z = self:encode(x)
	return self:decode(z)
end

function AutoEncoder:parameters(): { Types.Tensor }
	local params = {}
	Utils.collectParams(params, self._encoder)
	Utils.collectParams(params, self._decoder)
	return params
end

function AutoEncoder:train(mode: boolean?)
	Utils.trainAll(self._trainables, mode)
end

return setmetatable({
	new = function(cfg: AutoEncoderConfig): Types.Module<Types.Tensor, Types.Tensor>
		return AutoEncoder.new(cfg)
	end,
}, {
	__call = function(_, cfg: AutoEncoderConfig): Types.Module<Types.Tensor, Types.Tensor>
		return AutoEncoder.new(cfg)
	end,
})
