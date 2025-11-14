--!optimize 2
local Util = {}

local Types = require("./Types")

local function isModule(m: any): boolean
	return type(m) == "table" and type(m.forward) == "function" and type(m.parameters) == "function"
end
local function applyRecursive(m: any, fn: (any) -> ())
	if isModule(m) then
		fn(m)
	end
	if type(m) == "table" then
		if type(m._layers) == "table" then
			for _, sub in m._layers do
				applyRecursive(sub, fn)
			end
		end
		for _, v in m do
			if v ~= m and isModule(v) then
				applyRecursive(v, fn)
			end
		end
	end
end

--@param shape {number}
--@return number
function Util.sizeFromShape(shape: { number }): number
	task.desynchronize()
	local n = 1
	for _, d in shape do
		n *= d
	end
	task.synchronize()
	return n
end

function Util.train(m: Types.Module)
	applyRecursive(m, function(mm)
		(mm :: any)._train = true
	end)
end

function Util.eval(m: Types.Module)
	applyRecursive(m, function(mm)
		(mm :: any)._train = false
	end)
end

function Util.apply(m: Types.Module, fn: (Types.Module) -> ())
	applyRecursive(m, function(mm)
		fn(mm :: Types.Module)
	end)
end

function Util.to(m: Types.Module, dtype: Types.DType)
	applyRecursive(m, function(mm)
		for _, p in mm:parameters() do
			p._dtype = dtype
		end
	end)
end

function Util.shape(x: Types.Tensor): (number, number?)
    return x._shape[1], x._shape[2]
end

Util.ModelStats = require("@self/ModelStats")
Util.Visual2D = require("@self/Visual2D")
Util.Visual3D = require("@self/Visual3D")
Util.RunningNorm = require("@self/RunningNorm")
Util.Anomaly = require("@self/Anomaly")
Util.RNG = require("@self/RNG")
Util.Hooks = require("@self/Hooks")
Util.Parallel = require("@self/Parallel")
Util.GradStats = require("@self/GradStats")
Util.Buffer = require("@self/Buffer")

return Util
