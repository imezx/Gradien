--!optimize 2
local Types = require("../Types")
local Utils = require("./Utils")
local Tensor = require("../Tensor")
local GRU = require("../nn/GRU")
local LSTM = require("../nn/LSTM")
local RNN = require("../nn/RNN")
local Dropout = require("../nn/Dropout")
local LayerNorm = require("../nn/LayerNorm")
local Linear = require("../nn/Linear")

export type SequenceClassifierConfig = {
	inputDim: number,
	hiddenDim: number,
	numLayers: number?,
	numClasses: number,
	cell: "gru" | "lstm" | "rnn"?,
	dropout: number?,
	layerNorm: boolean?,
	finalActivation: Utils.ActivationName?,
}

type CellType = "gru" | "lstm" | "rnn"
type LSTMState = { h: Types.Tensor, c: Types.Tensor }

local function makeCell(cellType: CellType, inputDim: number, hiddenDim: number)
	if cellType == "lstm" then
		return LSTM(inputDim, hiddenDim)
	elseif cellType == "rnn" then
		return RNN(inputDim, hiddenDim)
	end
	return GRU(inputDim, hiddenDim)
end

local function sliceTimeStep(x: Types.Tensor, t: number, inputDim: number, batch: number): Types.Tensor
	local step = Tensor.slice(x, 2, t, t)
	return Tensor.reshape(step, { inputDim, batch })
end

local function initStates(self, batch: number, dtype: Types.DType, requiresGrad: boolean)
	local states = table.create(#self._cells)
	for i = 1, #self._cells do
		if self._cellType == "lstm" then
			states[i] = {
				h = Tensor.zeros({ self._hiddenDim, batch }, dtype, requiresGrad),
				c = Tensor.zeros({ self._hiddenDim, batch }, dtype, requiresGrad),
			}
		else
			states[i] = Tensor.zeros({ self._hiddenDim, batch }, dtype, requiresGrad)
		end
	end
	return states
end

local function forward(self, x: Types.Tensor): Types.Tensor
	assert(#x._shape == 3, "SequenceClassifier: expected input shape {features, time, batch}")
	local inputDim, timeSteps, batch = x._shape[1], x._shape[2], x._shape[3]
	assert(inputDim == self._inputDim, "SequenceClassifier: inputDim mismatch")
	local states = initStates(self, batch, x._dtype, x._requiresGrad)
	local last
	for t = 1, timeSteps do
		local step = sliceTimeStep(x, t, inputDim, batch)
		local layerInput: Types.Tensor = step
		for layerIdx, cell in ipairs(self._cells) do
			if self._cellType == "lstm" then
				local state = states[layerIdx] :: LSTMState
				local nextH, nextC = cell:forward(layerInput, state.h, state.c)
				states[layerIdx] = { h = nextH, c = nextC }
				layerInput = nextH
			else
				local prev = states[layerIdx] :: Types.Tensor
				local nextH = cell:forward(layerInput, prev)
				states[layerIdx] = nextH
				layerInput = nextH
			end
			local drop = self._dropoutBetween[layerIdx]
			if drop and layerIdx < #self._cells then
				layerInput = drop:forward(layerInput)
			end
		end
		last = layerInput
	end
	assert(last, "SequenceClassifier: sequence length must be >= 1")
	local out = last
	if self._norm then
		out = self._norm:forward(out)
	end
	if self._finalDropout then
		out = self._finalDropout:forward(out)
	end
	out = self._head:forward(out)
	if self._finalAct then
		out = self._finalAct:forward(out)
	end
	return out
end

local function parameters(self): { Types.Tensor }
	local params = {}
	for _, cell in ipairs(self._cells) do
		Utils.collectParams(params, cell)
	end
	if self._norm then
		Utils.collectParams(params, self._norm)
	end
	Utils.collectParams(params, self._head)
	return params
end

local function train(self, mode: boolean?)
	Utils.trainAll(self._trainables, mode)
end

local function SequenceClassifier(cfg: SequenceClassifierConfig): Types.Module<Types.Tensor, Types.Tensor>
	assert(cfg.inputDim and cfg.hiddenDim and cfg.numClasses, "SequenceClassifier: missing config fields")
	local numLayers = math.max(1, cfg.numLayers or 1)
	local cellType = (cfg.cell or "gru") :: CellType

	local layers = table.create(numLayers)
	local inDim = cfg.inputDim
	for i = 1, numLayers do
		layers[i] = makeCell(cellType, inDim, cfg.hiddenDim)
		inDim = cfg.hiddenDim
	end

	local dropoutBetween = {}
	local dropProb = math.clamp(cfg.dropout or 0, 0, 0.95)
	if dropProb > 0 and numLayers > 1 then
		for i = 1, numLayers - 1 do
			dropoutBetween[i] = Dropout(dropProb)
		end
	end
	local finalDropout = if dropProb > 0 then Dropout(dropProb) else nil
	local norm = if cfg.layerNorm then LayerNorm(cfg.hiddenDim) else nil
	local head = Linear(cfg.hiddenDim, cfg.numClasses)
	local finalAct = if cfg.finalActivation then Utils.makeActivationModule(cfg.finalActivation) else nil

	local trainables: { any } = {}
	for _, cell in ipairs(layers) do
		table.insert(trainables, cell)
	end
	for _, drop in ipairs(dropoutBetween) do
		table.insert(trainables, drop)
	end
	if finalDropout then
		table.insert(trainables, finalDropout)
	end
	if norm then
		table.insert(trainables, norm)
	end
	table.insert(trainables, head)
	if finalAct then
		table.insert(trainables, finalAct)
	end

	return {
		_inputDim = cfg.inputDim,
		_hiddenDim = cfg.hiddenDim,
		_cells = layers,
		_cellType = cellType,
		_dropoutBetween = dropoutBetween,
		_finalDropout = finalDropout,
		_norm = norm,
		_head = head,
		_finalAct = finalAct,
		_trainables = trainables,
		forward = forward,
		parameters = parameters,
		train = train,
	}
end

return {
	new = SequenceClassifier,
}
