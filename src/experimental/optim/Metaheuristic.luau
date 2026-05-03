--!native
--!optimize 2
-- Metaheuristic Optimizer for QIMHNN
local Types = require("../../Types")
local Tensor = require("../../Tensor")
local Util = require("../../Util")

local assert = Util.Assert

export type MetaheuristicConfig = {
	lr: number,
	swarmSize: number?,
	inertia: number?,
	cognitive: number?,
	social: number?,
	gradScale: number?,
	mutationRate: number?,
	mutationStrength: number?,
}

return function(params: { Types.Tensor }, config: MetaheuristicConfig): Types.Optimizer
	assert(config.lr and config.lr > 0, "MetaheuristicOptimizer: lr required")

	local lr = config.lr
	local _swarmSize = config.swarmSize or 10
	local inertia = config.inertia or 0.9
	local cognitive = config.cognitive or 1.5
	local social = config.social or 1.5
	local gradScale = config.gradScale or 0.1
	local mutationRate = config.mutationRate or 0.05
	local mutationStrength = config.mutationStrength or 0.01

	local velocities: { Types.Tensor } = {}
	local personalBest: { Types.Tensor } = {}
	local globalBest: { Types.Tensor } = {}
	local personalBestScore: { number } = {}
	local globalBestScore: { number } = {}

	for i, param in ipairs(params) do
		local v = Tensor.zeros(param._shapedtype, false)
		local pb = Tensor.zeros(param._shapedtype, false)
		local gb = Tensor.zeros(param._shapedtype, false)

		local pStore = param._storage
		local vStore = v._storage
		local pbStore = pb._storage
		local gbStore = gb._storage

		task.desynchronize()
		for j = 1, #pStore do
			vStore[j] = (math.random() * 2 - 1) * 0.001
			pbStore[j] = pStore[j]
			gbStore[j] = pStore[j]
		end
		task.synchronize()

		velocities[i] = v
		personalBest[i] = pb
		globalBest[i] = gb
		personalBestScore[i] = math.huge
		globalBestScore[i] = math.huge
	end
	return {
		step = function()
			for i, param in ipairs(params) do
				local grad = param._grad
				if grad then
					local gStore = grad._storage
					local gradNormSq = 0

					task.desynchronize()
					for j = 1, #gStore do
						local g = gStore[j]
						gradNormSq += g * g
					end
					task.synchronize()

					local score = gradNormSq
					if score < personalBestScore[i] then
						personalBestScore[i] = score
						local pb = personalBest[i]
						local pStore = param._storage
						local pbStore = pb._storage
						task.desynchronize()
						for j = 1, #pStore do
							pbStore[j] = pStore[j]
						end
						task.synchronize()
					end
					if score < globalBestScore[i] then
						globalBestScore[i] = score
						local gb = globalBest[i]
						local pStore = param._storage
						local gbStore = gb._storage
						task.desynchronize()
						for j = 1, #pStore do
							gbStore[j] = pStore[j]
						end
						task.synchronize()
					end
				end
			end
			for i, param in ipairs(params) do
				local grad = param._grad
				if grad then
					local v = velocities[i]
					local pb = personalBest[i]
					local gb = globalBest[i]

					local pStore = param._storage
					local gStore = grad._storage
					local vStore = v._storage
					local pbStore = pb._storage
					local gbStore = gb._storage
					task.desynchronize()
					for j = 1, #pStore do
						local r1 = math.random()
						local r2 = math.random()
						local g = gStore[j]

						local current = pStore[j]
						local vj = vStore[j]
						local pbj = pbStore[j]
						local gbj = gbStore[j]
						vj = inertia * vj
							+ cognitive * r1 * (pbj - current)
							+ social * r2 * (gbj - current)
							- gradScale * g

						local newVal = current + lr * vj
						if mutationRate > 0 and math.random() < mutationRate then
							newVal += (math.random() * 2 - 1) * mutationStrength
						end
						vStore[j] = vj
						pStore[j] = newVal
					end
					task.synchronize()
				end
			end
		end,
		zeroGrad = function()
			for _, param in ipairs(params) do
				local grad = param._grad
				if grad then
					local gStore = grad._storage
					task.desynchronize()
					for i = 1, #gStore do
						gStore[i] = 0
					end
					task.synchronize()
				end
			end
		end,
		getState = function()
			return nil
		end,
		setState = function(_self, _state)
			return
		end,
	}
end
