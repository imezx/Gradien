--!optimize 2
-- Quantum-Inspired Metaheuristic Neural Network (QIMHNN)
local Types = require("../../Types")
local Tensor = require("../../Tensor")
local Tape = require("../../autograd/Tape")
local Util = require("../../Util")
local Utils = require("../../models/Utils")
local LayerNorm = require("../../nn/LayerNorm")
local Dropout = require("../../nn/Dropout")
local Linear = require("../../nn/Linear")
local Sequential = require("../../nn/Sequential")

local assert = Util.Assert

export type QILayerConfig = {
	inDim: number,
	outDim: number,
	useSuperposition: boolean?,
	useInterference: boolean?,
	useEntanglement: boolean?,
	quantumAmplitude: number?, -- amp for |z|
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

	local interferenceAlpha = 0.1
	local entangleBeta = 0.1
	local eps = 1e-8

	local W_real = Tensor.zeros({ config.outDim, config.inDim }, "f64", true)
	local W_imag = Tensor.zeros({ config.outDim, config.inDim }, "f64", true) -- used when superposition OR interference enabled
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
			-- linear
			local z_real = Tape.matmul(self.W_real, x)
			local hasImag = (self.useSuperposition or self.useInterference)
			local z_imag: Types.Tensor? = nil
			if hasImag then
				z_imag = Tape.matmul(self.W_imag, x)
			end
			-- y_sup: superposition amplitude mix
			local y_sup = Tensor.zeros({ outF, B }, "f64", false)
			task.desynchronize()
			for i = 1, outF do
				for j = 1, B do
					local idx = (i - 1) * B + j
					local zr = z_real._storage[idx]
					if self.useSuperposition and z_imag then
						local zi = z_imag._storage[idx]
						local mag = math.sqrt(zr * zr + zi * zi + eps)
						y_sup._storage[idx] = zr + self.quantumAmplitude * mag
					else
						y_sup._storage[idx] = zr
					end
				end
			end
			task.synchronize()
			-- scale by 1 + alpha * cos(phase), phase = atan2(zi, zr)
			local y_int = Tensor.zeros({ outF, B }, "f64", y_sup._requiresGrad)
			task.desynchronize()
			for i = 1, outF do
				for j = 1, B do
					local idx = (i - 1) * B + j
					local scale = 1
					if self.useInterference then
						local zr = z_real._storage[idx]
						local zi = if z_imag then z_imag._storage[idx] else 0
						local phase = math.atan2(zi, zr)
						scale = 1 + interferenceAlpha * math.cos(phase)
					end
					y_int._storage[idx] = y_sup._storage[idx] * scale
				end
			end
			task.synchronize()
			local stored_y_int: Types.Tensor = nil
			if self.useEntanglement and self.entanglement then
				stored_y_int = Tensor.zeros({ outF, B }, "f64", false)
				task.desynchronize()
				for i = 1, #stored_y_int._storage do
					stored_y_int._storage[i] = y_int._storage[i]
				end
				task.synchronize()
			end
			task.synchronize()
			-- residual E @ y_int
			local y_preBias = y_int
			local entangled: Types.Tensor = nil
			if self.useEntanglement and self.entanglement then
				entangled = Tape.matmul(self.entanglement, y_int)
				task.desynchronize()
				for i = 1, #y_preBias._storage do
					y_preBias._storage[i] = y_preBias._storage[i] + entangleBeta * entangled._storage[i]
				end
				task.synchronize()
			end
			-- bias
			task.desynchronize()
			for i = 1, outF do
				local bi = self.b._storage[i]
				for j = 1, B do
					y_preBias._storage[(i - 1) * B + j] += bi
				end
			end
			task.synchronize()
			local stored_z_real = z_real
			local stored_z_imag = z_imag
			-- local stored_y_sup = y_sup
			local usedImag = hasImag
			local parents: { Types.Tensor } = { self.W_real, self.b, x }
			if self.useSuperposition or self.useInterference then
				table.insert(parents, self.W_imag)
			end
			if self.useEntanglement and self.entanglement then
				table.insert(parents, self.entanglement)
			end
			return Tape.makeNode(y_preBias, parents, function(gy: Types.Tensor)
				-- gy is dL/d(y_preBias)
				local outF = config.outDim
				local inF = config.inDim
				local B = gy._shape[2]
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
				-- y_preBias = y_int + beta * (E @ y_int)
				local gy_int = gy -- start with gy
				if self.useEntanglement and self.entanglement then
					-- dE += beta * gy @ y_int^T
					if self.entanglement._requiresGrad then
						if not self.entanglement._grad then
							self.entanglement._grad = Tensor.zeros({ outF, outF })
						end
						local yIntT: Types.Tensor = nil
						Tape.noGrad(function()
							yIntT = Tensor.transpose(stored_y_int)
						end)
						local dEnt = Tape.matmul(gy, yIntT)
						task.desynchronize()
						for i = 1, #self.entanglement._grad._storage do
							self.entanglement._grad._storage[i] += entangleBeta * dEnt._storage[i]
						end
						task.synchronize()
					end
					-- E: gy_int = gy + beta * (E^T @ gy)
					local ET
					Tape.noGrad(function()
						ET = Tensor.transpose(self.entanglement)
					end)
					local add = Tape.matmul(ET, gy)
					local tmp = Tensor.zeros({ outF, B }, "f64", false)
					task.desynchronize()
					for i = 1, #tmp._storage do
						tmp._storage[i] = gy._storage[i] + entangleBeta * add._storage[i]
					end
					task.synchronize()
					gy_int = tmp
				end
				local gz_real = Tensor.zeros({ outF, B }, "f64", false)
				local gz_imag = if usedImag then Tensor.zeros({ outF, B }, "f64", false) else nil
				task.desynchronize()
				for i = 1, outF do
					for j = 1, B do
						local idx = (i - 1) * B + j
						local zr = stored_z_real._storage[idx]
						local zi = if stored_z_imag then stored_z_imag._storage[idx] else 0
						local denom = zr * zr + zi * zi + eps
						local mag = math.sqrt(denom)

						local ysup = if self.useSuperposition then (zr + self.quantumAmplitude * mag) else zr

						local M = 1
						local phase = 0
						if self.useInterference then
							phase = math.atan2(zi, zr)
							M = 1 + interferenceAlpha * math.cos(phase)
						end
						-- gy wrt y_sup and wrt M
						local gyij = gy_int._storage[idx]
						local gy_sup = gyij * M
						local gM = gyij * ysup
						local gzR_phase, gzI_phase = 0, 0
						if self.useInterference then
							local gphase = -interferenceAlpha * gM * math.sin(phase)
							-- d phase/dzr = -zi/denom ; d phase/dzi = zr/denom
							gzR_phase = gphase * (-zi / denom)
							gzI_phase = gphase * (zr / denom)
						end
						-- superposition path
						local gzR_sup, gzI_sup = 0, 0
						if self.useSuperposition then
							gzR_sup = gy_sup * (1 + self.quantumAmplitude * (zr / mag))
							gzI_sup = gy_sup * (self.quantumAmplitude * (zi / mag))
						else
							gzR_sup = gy_sup
							gzI_sup = 0
						end
						local totalR = gzR_sup + gzR_phase
						gz_real._storage[idx] = gz_real._storage[idx] + totalR
						if gz_imag then
							local totalI = gzI_sup + gzI_phase
							gz_imag._storage[idx] = gz_imag._storage[idx] + totalI
						end
					end
				end
				task.synchronize()
				local xT
				Tape.noGrad(function()
					xT = Tensor.transpose(x)
				end)
				-- dW_real += gz_real @ x^T
				if self.W_real._requiresGrad then
					if not self.W_real._grad then
						self.W_real._grad = Tensor.zeros({ outF, inF })
					end
					local dW
					Tape.noGrad(function()
						dW = Tape.matmul(gz_real, xT)
					end)
					task.desynchronize()
					for i = 1, #self.W_real._grad._storage do
						self.W_real._grad._storage[i] += dW._storage[i]
					end
					task.synchronize()
				end
				-- dW_imag
				if gz_imag and self.W_imag._requiresGrad then
					if not self.W_imag._grad then
						self.W_imag._grad = Tensor.zeros({ outF, inF })
					end
					local dWi = Tape.matmul(gz_imag, xT)
					task.desynchronize()
					for i = 1, #self.W_imag._grad._storage do
						self.W_imag._grad._storage[i] += dWi._storage[i]
					end
					task.synchronize()
				end
				-- dx = W_real^T @ gz_real + (if used) W_imag^T @ gz_imag
				local dx: Types.Tensor = nil
				if x._requiresGrad then
					dx = Tensor.zeros({ inF, B }, "f64", false)
					local WT
					Tape.noGrad(function()
						WT = Tensor.transpose(self.W_real)
					end)
					local dxr
					Tape.noGrad(function()
						dxr = Tape.matmul(WT, gz_real)
					end)
					task.desynchronize()
					for i = 1, #dx._storage do
						dx._storage[i] = dxr._storage[i]
					end
					task.synchronize()
					if gz_imag then
						local WIT
						Tape.noGrad(function()
							WIT = Tensor.transpose(self.W_imag)
						end)
						local dxi
						Tape.noGrad(function()
							dxi = Tape.matmul(WIT, gz_imag)
						end)
						task.desynchronize()
						for i = 1, #dx._storage do
							dx._storage[i] += dxi._storage[i]
						end
						task.synchronize()
					end
					if not x._grad then
						x._grad = Tensor.zeros({ inF, B })
					end
					task.desynchronize()
					for i = 1, #x._grad._storage do
						x._grad._storage[i] += dx._storage[i]
					end
					task.synchronize()
				end
				return nil -- no grad
			end)
		end,
		parameters = function(self): { Types.Tensor }
			local params = { self.W_real, self.b }
			if self.useSuperposition or self.useInterference then
				table.insert(params, 2, self.W_imag)
			end
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
		act = function(_, x: Types.Tensor): Types.Tensor
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
