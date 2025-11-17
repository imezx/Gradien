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
	interferenceStrength: number?,
	entanglementStrength: number?,
	initScale: number?,
}

local function QILayer(config: QILayerConfig): Types.Module<Types.Tensor, Types.Tensor>
	assert(config.inDim and config.inDim > 0, "QILayer: inDim required")
	assert(config.outDim and config.outDim > 0, "QILayer: outDim required")

	local useSuperposition = config.useSuperposition ~= false
	local useInterference = config.useInterference ~= false
	local useEntanglement = config.useEntanglement ~= false
	local quantumAmplitude = config.quantumAmplitude or 0.1
	local alpha = config.interferenceStrength or 0.1
	local beta = config.entanglementStrength or 0.1
	local initScale = config.initScale or 0.1
	local eps = 1e-8
	local W_real = Tensor.zeros({ config.outDim, config.inDim }, "f64", true)
	local W_imag = Tensor.zeros({ config.outDim, config.inDim }, "f64", useSuperposition)
	local b = Tensor.zeros({ config.outDim, 1 }, "f64", true)
	local entanglement = if useEntanglement then Tensor.zeros({ config.outDim, config.outDim }, "f64", true) else nil
	task.desynchronize()
	local fanIn, fanOut = config.inDim, config.outDim
	local bound = initScale * math.sqrt(2 / (fanIn + fanOut))
	for i = 1, #W_real._storage do
		W_real._storage[i] = (math.random() * 2 - 1) * bound
	end
	for i = 1, #W_imag._storage do
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
		useSuperposition = useSuperposition,
		useInterference = useInterference,
		useEntanglement = useEntanglement,
		quantumAmplitude = quantumAmplitude,
		alpha = alpha,
		beta = beta,
		forward = function(self, x: Types.Tensor): Types.Tensor
			local B = x._shape[2]
			local outF = config.outDim
			-- linear
			local out_real = Tape.matmul(self.W_real, x) -- [outF, B]
			local out_imag: Types.Tensor? = nil
			if self.useSuperposition then
				out_imag = Tape.matmul(self.W_imag, x) -- [outF, B]
			end
			-- out_mix = amp * |z| + real
			local out_mix = Tensor.zeros({ outF, B }, "f64", x._requiresGrad or self.W_real._requiresGrad)
			task.desynchronize()
			if self.useSuperposition and out_imag then
				for i = 1, outF do
					for j = 1, B do
						local idx = (i - 1) * B + j
						local real = out_real._storage[idx]
						local imag = out_imag._storage[idx]
						local mag = math.sqrt(real * real + imag * imag + eps)
						out_mix._storage[idx] = self.quantumAmplitude * mag + real
					end
				end
			else
				for k = 1, #out_mix._storage do
					out_mix._storage[k] = out_real._storage[k]
				end
			end
			task.synchronize()
			-- f = 1 + alpha * cos(phase), phase = atan2(imag, real_pre)
			local out_after_interf = Tensor.zeros({ outF, B }, "f64", out_mix._requiresGrad)
			task.desynchronize()
			for i = 1, outF do
				for j = 1, B do
					local idx = (i - 1) * B + j
					local real_pre = out_real._storage[idx]
					local imag_pre = (out_imag and out_imag._storage[idx]) or 0
					local phase = math.atan2(imag_pre, real_pre)
					local f = 1 + self.alpha * math.cos(phase)
					out_after_interf._storage[idx] = out_mix._storage[idx] * f
				end
			end
			task.synchronize()
			-- y = z + beta * (E @ z)
			local out_before_ent = out_after_interf
			local out = out_before_ent
			local entangled: Types.Tensor? = nil
			if self.useEntanglement and self.entanglement then
				entangled = Tape.matmul(self.entanglement, out_before_ent)
				task.desynchronize()
				for k = 1, #out._storage do
					out._storage[k] = out._storage[k] + self.beta * entangled._storage[k]
				end
				task.synchronize()
			end
			-- bias
			task.desynchronize()
			for i = 1, outF do
				local bi = self.b._storage[i]
				for j = 1, B do
					out._storage[(i - 1) * B + j] += bi
				end
			end
			task.synchronize()
			local stored_out_real = out_real
			local stored_out_imag = out_imag
			local stored_out_mix = out_mix
			local stored_out_before_ent = out_before_ent

			local nodeInputs = { self.W_real, self.b, x }
			if self.useSuperposition then table.insert(nodeInputs, 2, self.W_imag) end
			if self.useEntanglement and self.entanglement then table.insert(nodeInputs, self.entanglement) end

			return Tape.makeNode(out, nodeInputs, function(gy: Types.Tensor)
				-- gy: dL/dy (post-bias, post-entanglement)
				-- dL/db = sum_j gy[i,j]
				if self.b._requiresGrad then
					if not self.b._grad then self.b._grad = Tensor.zeros({ outF, 1 }) end
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

				-- z = out_before_ent; y = z + beta * (E @ z)
				local gy_to_z = gy -- start with downstream grad
				if self.useEntanglement and self.entanglement and stored_out_before_ent then
					-- Add (E^T @ gy) * beta to gy_to_z
					local ET
					Tape.noGrad(function()
						ET = Tensor.transpose(self.entanglement)
					end)
					local add = Tape.matmul(ET, gy) -- [outF,B]
					task.desynchronize()
					for k = 1, #gy_to_z._storage do
						gy_to_z._storage[k] = gy_to_z._storage[k] + self.beta * add._storage[k]
					end
					task.synchronize()
					-- Grad wrt E: beta * (gy @ z^T)
					if self.entanglement._requiresGrad then
						if not self.entanglement._grad then self.entanglement._grad = Tensor.zeros({ outF, outF }) end
						local zT
						Tape.noGrad(function()
							zT = Tensor.transpose(stored_out_before_ent)
						end)
						local dE = Tape.matmul(gy, zT)
						task.desynchronize()
						for i = 1, #self.entanglement._grad._storage do
							self.entanglement._grad._storage[i] += self.beta * dE._storage[i]
						end
						task.synchronize()
					end
				end

				local g_real_total = Tensor.zeros({ outF, B }, "f64", false)
				local g_imag_total = Tensor.zeros({ outF, B }, "f64", false)

				task.desynchronize()
				for i = 1, outF do
					for j = 1, B do
						local idx = (i - 1) * B + j
						local real_pre = stored_out_real._storage[idx]
						local imag_pre = (stored_out_imag and stored_out_imag._storage[idx]) or 0
						local mag = math.sqrt(real_pre * real_pre + imag_pre * imag_pre + eps)

						-- Interference factor and derivatives
						local phase = math.atan2(imag_pre, real_pre)
						local cosP = math.cos(phase)
						local sinP = math.sin(phase)
						local f = 1 + self.alpha * cosP
						local denom = (real_pre * real_pre + imag_pre * imag_pre + eps)
						-- dphase/dreal = -imag/denom ; dphase/dimag = real/denom
						local dphase_dreal = -imag_pre / denom
						local dphase_dimag =  real_pre / denom
						-- df/d(real|imag) = -alpha * sin(phase) * dphase/d(real|imag)
						local df_dreal = -self.alpha * sinP * dphase_dreal
						local df_dimag = -self.alpha * sinP * dphase_dimag

						-- Product rule through interference: z = out_mix * f
						local g_down = gy_to_z._storage[idx]
						local out_mix_val = stored_out_mix._storage[idx]
						local g_outmix_from_interf = g_down * f
						local g_real_from_interf = g_down * out_mix_val * df_dreal
						local g_imag_from_interf = g_down * out_mix_val * df_dimag

						-- Superposition: out_mix = amp * mag + real
						local dreal_domix = self.quantumAmplitude * (real_pre / mag) + 1
						local dimag_domix = self.quantumAmplitude * (imag_pre / mag)

						local g_real = g_outmix_from_interf * dreal_domix + g_real_from_interf
						local g_imag = g_outmix_from_interf * dimag_domix + g_imag_from_interf

						g_real_total._storage[idx] = g_real
						g_imag_total._storage[idx] = g_imag
					end
				end
				task.synchronize()
				-- dW_real = g_real_total @ x^T
				if self.W_real._requiresGrad then
					if not self.W_real._grad then self.W_real._grad = Tensor.zeros({ outF, config.inDim }) end
					local xT
					Tape.noGrad(function()
						xT = Tensor.transpose(x)
					end)
					local dW = Tape.matmul(g_real_total, xT)
					task.desynchronize()
					for k = 1, #self.W_real._grad._storage do
						self.W_real._grad._storage[k] += dW._storage[k]
					end
					task.synchronize()
				end
				-- dW_imag = g_imag_total @ x^T (only if superposition used)
				if self.useSuperposition and self.W_imag._requiresGrad then
					if not self.W_imag._grad then self.W_imag._grad = Tensor.zeros({ outF, config.inDim }) end
					local xT
					Tape.noGrad(function()
						xT = Tensor.transpose(x)
					end)
					local dW_im = Tape.matmul(g_imag_total, xT)
					task.desynchronize()
					for k = 1, #self.W_imag._grad._storage do
						self.W_imag._grad._storage[k] += dW_im._storage[k]
					end
					task.synchronize()
				end
				-- dx = W_real^T @ g_real_total + (if superpos) W_imag^T @ g_imag_total
				if x._requiresGrad then
					if not x._grad then x._grad = Tensor.zeros({ config.inDim, B }) end
					local WT
					Tape.noGrad(function()
						WT = Tensor.transpose(self.W_real)
					end)
					local dx = Tape.matmul(WT, g_real_total)
					if self.useSuperposition then
						local WIT
						Tape.noGrad(function()
							WIT = Tensor.transpose(self.W_imag)
						end)
						local dx_im = Tape.matmul(WIT, g_imag_total)
						task.desynchronize()
						for k = 1, #dx._storage do
							dx._storage[k] += dx_im._storage[k]
						end
						task.synchronize()
					end
					task.desynchronize()
					for k = 1, #x._grad._storage do
						x._grad._storage[k] += dx._storage[k]
					end
					task.synchronize()
				end
			end)
		end,
		parameters = function(self): { Types.Tensor }
			local params = { self.W_real, self.b }
			if self.useSuperposition then table.insert(params, 2, self.W_imag) end
			if self.useEntanglement and self.entanglement then table.insert(params, self.entanglement) end
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
	interferenceStrength: number?,
	entanglementStrength: number?,
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
			interferenceStrength = config.interferenceStrength or 0.1,
			entanglementStrength = config.entanglementStrength or 0.1,
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
