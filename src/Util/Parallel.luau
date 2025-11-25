--!native
--!optimize 2
local Parallel = {}
function Parallel.with(fn: () -> ...any): ...any
	task.desynchronize()
	local res = { fn() }
	task.synchronize()
	return table.unpack(res)
end
function Parallel.withSafe(fn: () -> ...any): (boolean, ...any)
	task.desynchronize()
	local res = { pcall(fn) }
	task.synchronize()
	return table.unpack(res)
end
return Parallel
