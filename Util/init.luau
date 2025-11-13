--!optimize 2
local Util = {}

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

Util.ModelStats = require("@self/ModelStats")
Util.Visual2D = require("@self/Visual2D")
Util.Visual3D = require("@self/Visual3D")
Util.RunningNorm = require("@self/RunningNorm")
Util.Anomaly = require("@self/Anomaly")
Util.RNG = require("@self/RNG")
Util.Hooks = require("@self/Hooks")
Util.Parallel = require("@self/Parallel")

return Util
