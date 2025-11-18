--!optimize 2
-- Metaheuristic Optimizer for QIMHNN
local Types = require("../Types")
local Tensor = require("../Tensor")
local Util = require("../Util")

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
		local v = Tensor.zeros(param._shape, param._dtype, false)
		local pb = Tensor.zeros(param._shape, param._dtype, false)
		local gb = Tensor.zeros(param._shape, param._dtype, false)

		task.desynchronize()
		for j = 1, #param._storage do
			v._storage[j] = (math.random() * 2 - 1) * 0.001
			pb._storage[j] = param._storage[j]
			gb._storage[j] = param._storage[j]
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
				if param._grad then
					local gradNormSq = 0
					task.desynchronize()
					for j = 1, #param._grad._storage do
						local g = param._grad._storage[j]
						gradNormSq += g * g
					end
					task.synchronize()
					local gradNorm = math.sqrt(gradNormSq)

					if gradNorm < personalBestScore[i] then
						personalBestScore[i] = gradNorm
						local pb = personalBest[i]
						task.desynchronize()
						for j = 1, #param._storage do
							pb._storage[j] = param._storage[j]
						end
						task.synchronize()
					end

					if gradNorm < globalBestScore[i] then
						globalBestScore[i] = gradNorm
						local gb = globalBest[i]
						task.desynchronize()
						for j = 1, #param._storage do
							gb._storage[j] = param._storage[j]
						end
						task.synchronize()
					end
				end
			end
			for i, param in ipairs(params) do
				if param._grad then
					local v = velocities[i]
					local pb = personalBest[i]
					local gb = globalBest[i]

					task.desynchronize()
					for j = 1, #param._storage do
						local r1 = math.random()
						local r2 = math.random()
						local grad = param._grad._storage[j]

						local current = param._storage[j]
						local vj = v._storage[j]
						local pbj = pb._storage[j]
						local gbj = gb._storage[j]

						vj = inertia * vj
							+ cognitive * r1 * (pbj - current)
							+ social * r2 * (gbj - current)
							- gradScale * grad

						local newVal = current + lr * vj

						if math.random() < mutationRate then
							newVal += (math.random() * 2 - 1) * mutationStrength
						end

						v._storage[j] = vj
						param._storage[j] = newVal
					end
					task.synchronize()
				end
			end
		end,
		zeroGrad = function()
			for _, param in ipairs(params) do
				if param._grad then
					task.desynchronize()
					for i = 1, #param._grad._storage do
						param._grad._storage[i] = 0
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
