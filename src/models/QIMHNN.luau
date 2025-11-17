--!optimize 2
-- Quantum-Inspired Metaheuristic Neural Network (QIMHNN)
local Types = require("../Types")
local Tensor = require("../Tensor")
local Tape = require("../autograd/Tape")
local Util = require("../Util")
local Utils = require("./Utils")
local LayerNorm = require("../nn/LayerNorm")
local Dropout = require("../nn/Dropout")
local Linear = require("../nn/Linear")
local Sequential = require("../nn/Sequential")

local assert = Util.Assert

export type QILayerConfig = {
	inDim: number,
	outDim: number,
	useSuperposition: boolean?,
	useInterference: boolean?,
	useEntanglement: boolean?,
	quantumAmplitude: number?,
	initScale: number?,
}

local function QILayer(config: QILayerConfig): Types.Module<Types.Tensor, Types.Tensor>
	assert(config.inDim and config.inDim > 0, "QILayer: inDim required")
	assert(config.outDim and config.outDim > 0, "QILayer: outDim required")

	local useSuperposition = config.useSuperposition ~= false
	local useInterference = config.useInterference ~= false
	local useEntanglement = config.useEntanglement ~= false
	local quantumAmplitude = config.quantumAmplitude or 0.1
	local initScale = config.initScale or 0.1

	local W_real = Tensor.zeros({ config.outDim, config.inDim }, "f64", true)
	local W_imag = Tensor.zeros({ config.outDim, config.inDim }, "f64", true)
	local b = Tensor.zeros({ config.outDim, 1 }, "f64", true)

	local entanglement = if useEntanglement then Tensor.zeros({ config.outDim, config.outDim }, "f64", true) else nil

	task.desynchronize()
	local fanIn, fanOut = config.inDim, config.outDim
	local bound = initScale * math.sqrt(2 / (fanIn + fanOut))
	for i = 1, #W_real._storage do
		W_real._storage[i] = (math.random() * 2 - 1) * bound
		W_imag._storage[i] = (math.random() * 2 - 1) * bound * 0.5
	end
	if entanglement then
		for i = 1, #entanglement._storage do
			entanglement._storage[i] = (math.random() * 2 - 1) * initScale * 0.1
		end
	end
	task.synchronize()
	return {
		W_real = W_real,
		W_imag = W_imag,
		b = b,
		entanglement = entanglement,
		quantumAmplitude = quantumAmplitude,
		useSuperposition = useSuperposition,
		useInterference = useInterference,
		useEntanglement = useEntanglement,
		forward = function(self, x: Types.Tensor): Types.Tensor
			local B = x._shape[2]
			local outF = config.outDim

			local out_real = Tape.matmul(self.W_real, x)
			local out = Tensor.zeros({ outF, B }, "f64", x._requiresGrad or self.W_real._requiresGrad)

			if useSuperposition then
				local out_imag = Tape.matmul(self.W_imag, x)
				-- |z|^2 = real^2 + imag^2
				task.desynchronize()
				for i = 1, outF do
					for j = 1, B do
						local idx = (i - 1) * B + j
						local real = out_real._storage[idx]
						local imag = out_imag._storage[idx]
						out._storage[idx] = math.sqrt(real * real + imag * imag + 1e-8) * self.quantumAmplitude + real
					end
				end
				task.synchronize()
			else
				task.desynchronize()
				for i = 1, #out._storage do
					out._storage[i] = out_real._storage[i]
				end
				task.synchronize()
			end
			if useInterference then
				task.desynchronize()
				for i = 1, outF do
					for j = 1, B do
						local idx = (i - 1) * B + j
						local phase = math.atan2(
							if useSuperposition
								then self.W_imag._storage[math.floor((i - 1) / outF) * config.inDim + 1]
								else 0,
							out._storage[idx]
						)
						out._storage[idx] = out._storage[idx] * (1 + 0.1 * math.cos(phase))
					end
				end
				task.synchronize()
			end
			if useEntanglement and self.entanglement then
				local entangled = Tape.matmul(self.entanglement, out)
				task.desynchronize()
				for i = 1, #out._storage do
					out._storage[i] = out._storage[i] + 0.1 * entangled._storage[i]
				end
				task.synchronize()
			end
			-- Add bias
			task.desynchronize()
			for i = 1, outF do
				local bi = self.b._storage[i]
				for j = 1, B do
					out._storage[(i - 1) * B + j] += bi
				end
			end
			task.synchronize()
			return Tape.makeNode(out, { self.W_real, self.W_imag, self.b, x }, function(gy: Types.Tensor)
				if self.W_real._requiresGrad then
					if not self.W_real._grad then
						self.W_real._grad = Tensor.zeros({ outF, config.inDim })
					end
					local xT
					Tape.noGrad(function()
						xT = Tensor.transpose(x)
					end)
					local dW = Tape.matmul(gy, xT)
					task.desynchronize()
					for i = 1, #self.W_real._grad._storage do
						self.W_real._grad._storage[i] += dW._storage[i]
					end
					task.synchronize()
				end
				if self.b._requiresGrad then
					if not self.b._grad then
						self.b._grad = Tensor.zeros({ outF, 1 })
					end
					task.desynchronize()
					for i = 1, outF do
						local acc = 0
						for j = 1, B do
							acc += gy._storage[(i - 1) * B + j]
						end
						self.b._grad._storage[i] += acc
					end
					task.synchronize()
				end
			end)
		end,
		parameters = function(self): { Types.Tensor }
			local params = { self.W_real, self.W_imag, self.b }
			if self.entanglement then
				table.insert(params, self.entanglement)
			end
			return params
		end,
	}
end

export type QIMHNNConfig = {
	inputDim: number,
	outputDim: number,
	hiddenDims: { number }?,
	activation: Utils.ActivationName?,
	finalActivation: Utils.ActivationName?,
	useSuperposition: boolean?,
	useInterference: boolean?,
	useEntanglement: boolean?,
	quantumAmplitude: number?,
	dropout: number?,
	layerNorm: boolean?,
}

return function(config: QIMHNNConfig): Types.Module<Types.Tensor, Types.Tensor>
	assert(config.inputDim, "QIMHNN: inputDim is required")
	assert(config.outputDim, "QIMHNN: outputDim is required")

	local hidden = if config.hiddenDims and #config.hiddenDims > 0
		then table.clone(config.hiddenDims)
		else { math.max(config.inputDim, config.outputDim) }
	local dropProb = math.clamp(config.dropout or 0, 0, 0.95)
	local layers = {}
	local trainables: { any } = {}
	local inDim = config.inputDim

	for _, hiddenDim in ipairs(hidden) do
		assert(hiddenDim > 0, "QIMHNN: hidden dimensions must be positive")
		local qiLayer = QILayer({
			inDim = inDim,
			outDim = hiddenDim,
			useSuperposition = config.useSuperposition,
			useInterference = config.useInterference,
			useEntanglement = config.useEntanglement,
			quantumAmplitude = config.quantumAmplitude or 0.1,
		})
		table.insert(layers, qiLayer)
		table.insert(trainables, qiLayer)
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
	local head = Linear(inDim, config.outputDim)
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
