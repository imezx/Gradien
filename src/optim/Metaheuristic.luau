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
	mutationRate: number?,
	mutationStrength: number?,
}

return function(params: { Types.Tensor }, config: MetaheuristicConfig): Types.Optimizer
	assert(config.lr and config.lr > 0, "MetaheuristicOptimizer: lr required")
	local lr = config.lr
	local swarmSize = config.swarmSize or 10
	local inertia = config.inertia or 0.9
	local cognitive = config.cognitive or 1.5
	local social = config.social or 1.5
	local mutationRate = config.mutationRate or 0.1
	local mutationStrength = config.mutationStrength or 0.01

	local velocities: { { Types.Tensor } } = {}
	local personalBest: { { Types.Tensor } } = {}
	local globalBest: { Types.Tensor } = {}

	for i = 1, swarmSize do
		local velocity: { Types.Tensor } = {}
		local pbest: { Types.Tensor } = {}

		for _, param in ipairs(params) do
			local v = Tensor.zeros(param._shape, param._dtype, false)
			local pb = Tensor.zeros(param._shape, param._dtype, false)

			task.desynchronize()
			for j = 1, #param._storage do
				v._storage[j] = (math.random() * 2 - 1) * 0.001
				pb._storage[j] = param._storage[j]
			end
			task.synchronize()

			table.insert(velocity, v)
			table.insert(pbest, pb)
		end

		table.insert(velocities, velocity)
		table.insert(personalBest, pbest)
	end
	for _, param in ipairs(params) do
		local gb = Tensor.zeros(param._shape, param._dtype, false)
		task.desynchronize()
		for j = 1, #param._storage do
			gb._storage[j] = param._storage[j]
		end
		task.synchronize()
		table.insert(globalBest, gb)
	end
	return {
		step = function()
			for i, param in ipairs(params) do
				if param._grad then
					local particleIdx = ((i - 1) % swarmSize) + 1
					local v = velocities[particleIdx][i]
					local pb = personalBest[particleIdx][i]
					local gb = globalBest[i]

					task.desynchronize()
					for j = 1, #param._storage do
						local r1 = math.random()
						local r2 = math.random()
						local grad = param._grad._storage[j]

						v._storage[j] = inertia * v._storage[j]
							+ cognitive * r1 * (pb._storage[j] - param._storage[j])
							+ social * r2 * (gb._storage[j] - param._storage[j])
							- 0.1 * grad -- Incorporate gradient

						param._storage[j] = param._storage[j] + lr * v._storage[j]

						-- mutation
						if math.random() < mutationRate then
							param._storage[j] = param._storage[j] + (math.random() * 2 - 1) * mutationStrength
						end

						if math.abs(param._storage[j] - pb._storage[j]) < 0.01 then
							pb._storage[j] = param._storage[j]
						end

						if math.abs(param._storage[j] - gb._storage[j]) < 0.01 then
							gb._storage[j] = param._storage[j]
						end
					end
					task.synchronize()
				else
					task.desynchronize()
					for j = 1, #param._storage do
						param._storage[j] = param._storage[j] - lr * (param._grad and param._grad._storage[j] or 0)
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
		setState = function(_self, state) end,
	}
end
