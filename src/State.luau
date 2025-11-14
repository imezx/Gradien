--!optimize 2
local Types = require("./Types")
local State = {}

--@param params { Types.Tensor } -- list of Tensors from module:parameters()
--@return Snapshot
function State.dump(params: { Types.Tensor }): Types.Snapshot
	task.desynchronize()
	local n = #params
	local out: Types.Snapshot = table.create(n)
	for i = 1, n do
		local t = params[i]
		assert(t and t._storage and t._shape, ("State.dump: expected Tensor at params[%d]"):format(i))
		local shp = table.clone(t._shape)
		local dat = table.create(#t._storage, 0)
		for k = 1, #t._storage do
			dat[k] = t._storage[k]
		end
		out[i] = { shape = shp, dtype = t._dtype or "f32", data = dat }
	end
	task.synchronize()
	return out
end

--@param params { Types.Tensor } -- list of Tensors from module:parameters()
--@param snap Snapshot? -- snapshot previously produced by State.dump (or nil)
--@return boolean -- true if anything was loaded
function State.load(params: { Types.Tensor }, snap: Types.Snapshot): boolean
	if snap == nil then
		return false
	end
	task.desynchronize()
	local changed = false
	for i = 1, math.min(#params, #snap) do
		local t = params[i]
		local s = snap[i]
		local data = s and (s.data or s.values)
		if t and data then
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

function State.dumpOptimizer(optim: Types.Optimizer): Types.OptimizerState?
	if optim and type(optim.getState) == "function" then
		return optim:getState()
	end
	return nil
end

function State.loadOptimizer(optim: Types.Optimizer, snap: Types.OptimizerState?): ()
	if not snap then
		return
	end
	if optim and type(optim.setState) == "function" then
		optim:setState(snap)
	end
end

function State.dumpTrainer(trainer: Types.Trainer): Types.TrainerSnapshot
	return {
		model = State.dumpModel(trainer.model),
		optimizer = State.dumpOptimizer(trainer.optimizer),
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
		State.loadOptimizer(trainer.optimizer, snap.optimizer)
	end
	trainer.step = snap.step or trainer.step or 0
	trainer.epoch = snap.epoch or trainer.epoch or 0
	trainer.bestMetric = snap.bestMetric
end

return State
