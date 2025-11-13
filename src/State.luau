--!optimize 2
local Types = require("./Types")
local State = {}

export type SnapshotTensor = {
	shape: { number },
	dtype: Types.DType,
	data: { number },
}

export type Snapshot = { SnapshotTensor }

--@param params {any} -- list of Tensors from module:parameters()
--@return Snapshot
function State.dump(params: { Types.Tensor }): { SnapshotTensor }
	task.desynchronize()
	local n = #params
	local out: { SnapshotTensor } = table.create(n)
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

--@param params { Tensor } -- list of Tensors from module:parameters()
--@param snap Snapshot|nil -- snapshot previously produced by State.dump (or nil)
--@return boolean -- true if anything was loaded
function State.load(params: { Types.Tensor }, snap: { SnapshotTensor }): boolean
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

function State.dumpModel(model: Types.Module): { SnapshotTensor }
	return State.dump(model:parameters())
end

function State.loadModel(model: Types.Module, snap: { SnapshotTensor }): boolean
	return State.load(model:parameters(), snap)
end

return State
