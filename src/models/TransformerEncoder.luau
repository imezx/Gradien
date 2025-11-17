--!optimize 2
local Types = require("../Types")
local Utils = require("./Utils")
local Linear = require("../nn/Linear")
local LayerNorm = require("../nn/LayerNorm")
local Dropout = require("../nn/Dropout")
local Sequential = require("../nn/Sequential")
local Attention = require("../nn/Attention")

export type TransformerEncoderConfig = {
	embedDim: number,
	numHeads: number,
	mlpDim: number?,
	numLayers: number?,
	inputDim: number?,
	dropout: number?,
	attentionDropout: number?,
	activation: Utils.ActivationName?,
	finalLayerNorm: boolean?,
	weightInit: ((number, number) -> number)?,
}

type Block = {
	norm1: Types.Module<Types.Tensor, Types.Tensor>,
	attn: Types.Module<Types.Tensor, Types.Tensor>,
	drop1: Types.Module<Types.Tensor, Types.Tensor>?,
	norm2: Types.Module<Types.Tensor, Types.Tensor>,
	ff: Types.Module<Types.Tensor, Types.Tensor>,
	trainables: { any },
}

local function buildFeedForward(
	embedDim: number,
	mlpDim: number,
	activation: Utils.ActivationName?,
	dropProb: number,
	init: ((number, number) -> number)?
): (Types.Module<Types.Tensor, Types.Tensor>, { any })
	local layers = {}
	local trainables: { any } = {}
	local function insert(m)
		table.insert(layers, m)
		table.insert(trainables, m)
	end
	insert(Linear(embedDim, mlpDim, init))
	insert(Utils.makeActivationModule(activation or "gelu"))
	if dropProb > 0 then
		insert(Dropout(dropProb))
	end
	insert(Linear(mlpDim, embedDim, init))
	if dropProb > 0 then
		insert(Dropout(dropProb))
	end
	return Sequential(layers), trainables
end

local function buildBlock(embedDim: number, cfg: TransformerEncoderConfig): Block
	local drop = math.clamp(cfg.dropout or 0, 0, 0.95)
	local norm1 = LayerNorm(embedDim)
	local attn = Attention(embedDim, cfg.numHeads, {
		attentionDropout = cfg.attentionDropout or drop,
	})
	local drop1 = if drop > 0 then Dropout(drop) else nil
	local norm2 = LayerNorm(embedDim)
	local mlpDim = cfg.mlpDim or (embedDim * 4)
	local ff, ffTrainables = buildFeedForward(embedDim, mlpDim, cfg.activation, drop, cfg.weightInit)
	local trainables: { any } = { norm1, attn, norm2 }
	if drop1 then
		table.insert(trainables, drop1)
	end
	for _, m in ipairs(ffTrainables) do
		table.insert(trainables, m)
	end

	return {
		norm1 = norm1,
		attn = attn,
		drop1 = drop1,
		norm2 = norm2,
		ff = ff,
		trainables = trainables,
	}
end

local function buildEncoder(cfg: TransformerEncoderConfig): Types.Module<Types.Tensor, Types.Tensor>
	assert(cfg.embedDim and cfg.numHeads, "TransformerEncoder: embedDim/numHeads required")
	local numLayers = math.max(1, cfg.numLayers or 4)
	local blocks = table.create(numLayers)
	local trainables: { any } = {}
	for i = 1, numLayers do
		local block = buildBlock(cfg.embedDim, cfg)
		blocks[i] = block
		for _, m in ipairs(block.trainables) do
			table.insert(trainables, m)
		end
	end

	local inputProj = if cfg.inputDim and cfg.inputDim ~= cfg.embedDim
		then Linear(cfg.inputDim, cfg.embedDim, cfg.weightInit)
		else nil
	local finalNorm = if cfg.finalLayerNorm ~= false then LayerNorm(cfg.embedDim) else nil
	if inputProj then
		table.insert(trainables, inputProj)
	end
	if finalNorm then
		table.insert(trainables, finalNorm)
	end

	local function forward(_, x: Types.Tensor): Types.Tensor
		local h = if inputProj then inputProj:forward(x) else x
		for _, block in ipairs(blocks) do
			local y = block.norm1:forward(h)
			y = block.attn:forward(y)
			if block.drop1 then
				y = block.drop1:forward(y)
			end
			h = Utils.addTensors(h, y)

			local z = block.norm2:forward(h)
			z = block.ff:forward(z)
			h = Utils.addTensors(h, z)
		end
		if finalNorm then
			h = finalNorm:forward(h)
		end
		return h
	end

	local function parameters(_): { Types.Tensor }
		local params = {}
		if inputProj then
			Utils.collectParams(params, inputProj)
		end
		for _, block in ipairs(blocks) do
			Utils.collectParams(params, block.norm1)
			Utils.collectParams(params, block.attn)
			if block.drop1 then
				Utils.collectParams(params, block.drop1)
			end
			Utils.collectParams(params, block.norm2)
			Utils.collectParams(params, block.ff)
		end
		if finalNorm then
			Utils.collectParams(params, finalNorm)
		end
		return params
	end

	local function train(_, mode: boolean?)
		Utils.trainAll(trainables, mode)
	end

	return {
		forward = forward,
		parameters = parameters,
		train = train,
	}
end

return {
	new = buildEncoder,
}
