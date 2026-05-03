--!native
--!optimize 2
local Types = require("./Types")
local Util = require("./Util")
local State = {}

local assert = Util.Assert

local function mapParamsToIndices(params: { Types.Tensor }): { [Types.Tensor]: string }
	local map = {}
	for i, p in params do
		map[p] = `{i}`
	end
	return map
end

--@param params { Types.Tensor }
--@return Snapshot
function State.dump(params: { Types.Tensor }): Types.Snapshot
	task.desynchronize()
	local n = #params
	local out: Types.Snapshot = table.create(n)
	for i = 1, n do
		local t = params[i]
		assert(t and t._storage and t._shape, `State.dump: expected Tensor at params[{i}]`)
		local shp = table.clone(t._shape)
		local dat = table.create(#t._storage, 0)
		for k = 1, #t._storage do
			dat[k] = t._storage[k]
		end
		out[i] = { shape = shp, data = dat }
	end
	task.synchronize()
	return out
end

function State.load(params: { Types.Tensor }, snap: Types.Snapshot): boolean
	if snap == nil then
		return false
	end
	task.desynchronize()
	local changed = false
	if #params ~= #snap then
		warn(`Gradien: Model parameter count mismatch (Model: {#params}, Snapshot: {#snap})`)
	end
	for i = 1, math.min(#params, #snap) do
		local t = params[i]
		local s = snap[i]
		local data = s and (s.data or s.values)
		if t and data then
			if #t._storage ~= #data then
				warn(`Gradien: Layer {i} size mismatch (Model: {#t._storage}, Snapshot: {#data})`)
			end
			local n = math.min(#t._storage, #data)
			for k = 1, n do
				t._storage[k] = data[k] :: number
			end
			changed = true
		end
	end
	task.synchronize()
	return changed
end

function State.dumpModel(model: Types.Module): Types.Snapshot
	return State.dump(model:parameters())
end

function State.loadModel(model: Types.Module, snap: Types.Snapshot): boolean
	return State.load(model:parameters(), snap)
end

-- Serializes optimizer state by converting Tensor Keys -> Index Strings
function State.dumpOptimizer(optim: Types.Optimizer, modelParams: { Types.Tensor }): Types.OptimizerState?
	if optim and type(optim.getState) == "function" then
		local rawState = optim:getState()
		if not rawState then
			return nil
		end
		local serialized = table.clone(rawState)
		local paramMap = mapParamsToIndices(modelParams)
		for k, v in rawState do
			if type(v) == "table" then
				local isTensorMap = false
				local newTable = {}
				for tk, tv in v do
					if type(tk) == "table" and tk._storage then
						isTensorMap = true
						local idx = paramMap[tk]
						if idx then
							newTable[idx] = tv
						end
					else
						newTable[tk] = tv
					end
				end
				if isTensorMap then
					serialized[k] = newTable
				end
			end
		end
		return serialized
	end
	return nil
end

function State.loadOptimizer(optim: Types.Optimizer, snap: Types.OptimizerState?, modelParams: { Types.Tensor }): ()
	if not snap then
		return
	end
	if optim and type(optim.setState) == "function" then
		local restored = table.clone(snap)
		for k, v in snap do
			if type(v) == "table" then
				local reconstructed = {}
				local wasParamMap = false
				for idxStr, val in v do
					local idx = tonumber(idxStr)
					if idx and modelParams[idx] then
						wasParamMap = true
						reconstructed[modelParams[idx]] = val
					else
						reconstructed[idxStr] = val
					end
				end
				if wasParamMap then
					restored[k] = reconstructed
				end
			end
		end
		optim:setState(restored)
	end
end

function State.dumpTrainer(trainer: Types.Trainer): Types.TrainerSnapshot
	local params = trainer.model:parameters()
	return {
		model = State.dumpModel(trainer.model),
		optimizer = State.dumpOptimizer(trainer.optimizer, params),
		step = trainer.step or nil,
		epoch = trainer.epoch or nil,
		bestMetric = trainer.bestMetric,
	}
end

function State.loadTrainer(trainer: Types.Trainer, snap: Types.TrainerSnapshot)
	if not snap then
		error("State.loadTrainer: no snapshot provided")
	end
	State.loadModel(trainer.model, snap.model)
	if snap.optimizer then
		local params = trainer.model:parameters()
		State.loadOptimizer(trainer.optimizer, snap.optimizer, params)
	end
	trainer.step = snap.step or trainer.step or 0
	trainer.epoch = snap.epoch or trainer.epoch or 0
	trainer.bestMetric = snap.bestMetric
end

function State.toBuffer(snap: Types.Snapshot): buffer
	return Util.Buffer.write(snap)
end

function State.fromBuffer(b: buffer): Types.Snapshot?
	return Util.Buffer.read(b)
end

return State
