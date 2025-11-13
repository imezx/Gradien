--!optimize 2
local Initializer = {}
function Initializer.xavierUniform(W)
	task.desynchronize()
	local fanIn = W._shape[2]
	local fanOut = W._shape[1]
	local bound = math.sqrt(6 / (fanIn + fanOut))
	for i = 1, #W._storage do
		W._storage[i] = (math.random() * 2 - 1) * bound
	end
	task.synchronize()
end
function Initializer.kaimingUniform(W)
	task.desynchronize()
	local fanIn = W._shape[2]
	local bound = math.sqrt(6 / fanIn)
	for i = 1, #W._storage do
		W._storage[i] = (math.random() * 2 - 1) * bound
	end
	task.synchronize()
end
function Initializer.zeros(B)
	task.desynchronize()
	for i = 1, #B._storage do
		B._storage[i] = 0
	end
	task.synchronize()
end
return Initializer
