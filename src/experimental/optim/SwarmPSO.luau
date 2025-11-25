--!native
--!optimize 2
-- multi-particle PSO Optimizer for QIMHNN
local Types = require("../../Types")
local Tensor = require("../../Tensor")
local Util = require("../../Util")

local assert = Util.Assert

export type SwarmPSOConfig = {
	swarmSize: number,
	inertia: number?,
	cognitive: number?,
	social: number?,
	lr: number?,
	mutationRate: number?,
	mutationStrength: number?,
	evalFn: (number) -> number,
}

local function copyTensorData(src: Types.Tensor, dst: Types.Tensor)
	assert(#src._storage == #dst._storage, "SwarmPSO: tensor size mismatch in copyTensorData")
	task.desynchronize()
	for i = 1, #src._storage do
		dst._storage[i] = src._storage[i]
	end
	task.synchronize()
end
local function swapTensorStorage(a: Types.Tensor, b: Types.Tensor)
	local tmp = a._storage
	a._storage = b._storage
	b._storage = tmp
end

return function(params: { Types.Tensor }, config: SwarmPSOConfig): Types.Optimizer
	assert(config and type(config) == "table", "SwarmPSO: config table required")
	assert(config.swarmSize and config.swarmSize > 0, "SwarmPSO: swarmSize > 0 required")
	assert(config.evalFn, "SwarmPSO: evalFn callback is required")

	local swarmSize = config.swarmSize
	local inertia = config.inertia or 0.7
	local cognitive = config.cognitive or 1.5
	local social = config.social or 1.5
	local lr = config.lr or 1.0
	local mutationRate = config.mutationRate or 0.0
	local mutationStrength = config.mutationStrength or 0.0
	local evalFn = config.evalFn

	local positions: { { Types.Tensor } } = {}
	local velocities: { { Types.Tensor } } = {}
	local pbest: { { Types.Tensor } } = {}
	local pbestScore: { number } = {}

	local gbest: { Types.Tensor } = {}
	local gbestScore = math.huge

	for p = 1, swarmSize do
		local posForParticle: { Types.Tensor } = {}
		local velForParticle: { Types.Tensor } = {}
		local pbestForParticle: { Types.Tensor } = {}
		for _, param in ipairs(params) do
			local pos = Tensor.zeros(param._shape, param._dtype, false)
			copyTensorData(param, pos)

			local vel = Tensor.zeros(param._shape, param._dtype, false)
			task.desynchronize()
			for j = 1, #vel._storage do
				vel._storage[j] = (math.random() * 2 - 1) * 0.001
			end
			task.synchronize()

			local pb = Tensor.zeros(param._shape, param._dtype, false)
			copyTensorData(param, pb)

			table.insert(posForParticle, pos)
			table.insert(velForParticle, vel)
			table.insert(pbestForParticle, pb)
		end
		positions[p] = posForParticle
		velocities[p] = velForParticle
		pbest[p] = pbestForParticle
		pbestScore[p] = math.huge
	end
	for _, param in ipairs(params) do
		local gb = Tensor.zeros(param._shape, param._dtype, false)
		copyTensorData(param, gb)
		table.insert(gbest, gb)
	end
	for i, param in ipairs(params) do
		param._storage = gbest[i]._storage
	end
	return {
		step = function()
			for p = 1, swarmSize do
				local posForParticle = positions[p]
				for i, param in ipairs(params) do
					local pos = posForParticle[i]
					swapTensorStorage(param, pos)
				end
				local loss = evalFn(p)
				assert(type(loss) == "number", "SwarmPSO: evalFn must return a number (loss)")
				for i, param in ipairs(params) do
					local pos = posForParticle[i]
					swapTensorStorage(param, pos)
				end
				if loss < pbestScore[p] then
					pbestScore[p] = loss
					local pbForParticle = pbest[p]
					for i = 1, #params do
						local pos = posForParticle[i]
						local pb = pbForParticle[i]
						copyTensorData(pos, pb)
					end
				end
				if loss < gbestScore then
					gbestScore = loss
					for i = 1, #params do
						local pos = posForParticle[i]
						local gb = gbest[i]
						copyTensorData(pos, gb)
					end
				end
			end
			for p = 1, swarmSize do
				local posForParticle = positions[p]
				local velForParticle = velocities[p]
				local pbForParticle = pbest[p]
				for i = 1, #params do
					local pos = posForParticle[i]
					local vel = velForParticle[i]
					local pb = pbForParticle[i]
					local gb = gbest[i]
					task.desynchronize()
					for j = 1, #pos._storage do
						local x = pos._storage[j]
						local vj = vel._storage[j]
						local pbestVal = pb._storage[j]
						local gbestVal = gb._storage[j]
						local r1 = math.random()
						local r2 = math.random()

						vj = inertia * vj + cognitive * r1 * (pbestVal - x) + social * r2 * (gbestVal - x)
						vj *= lr

						local newX = x + vj
						if mutationRate > 0 and math.random() < mutationRate then
							newX += (math.random() * 2 - 1) * mutationStrength
						end
						vel._storage[j] = vj
						pos._storage[j] = newX
					end
					task.synchronize()
				end
			end
			for i, param in ipairs(params) do
				param._storage = gbest[i]._storage
				if param._grad then
					task.desynchronize()
					for j = 1, #param._grad._storage do
						param._grad._storage[j] = 0
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
