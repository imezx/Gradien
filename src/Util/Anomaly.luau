--!native
--!optimize 2
local Types = require("../Types")
local Anomaly = {}
function Anomaly.hasBadValues(t: Types.Tensor)
	for i = 1, #t._storage do
		local v = t._storage[i]
		if v ~= v or v == math.huge or v == -math.huge then
			return true
		end
	end
	return false
end
function Anomaly.hasNaN(t: Types.Tensor)
	for i = 1, #t._storage do
		local v = t._storage[i]
		if v ~= v then
			return true
		end
	end
	return false
end
function Anomaly.hasInf(t: Types.Tensor)
	for i = 1, #t._storage do
		local v = t._storage[i]
		if v == math.huge or v == -math.huge then
			return true
		end
	end
	return false
end
return Anomaly
