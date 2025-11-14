--!optimize 2
local GC = {}
function GC.clipValue(params, clip)
	task.desynchronize()
	for _, p in params do
		if p._grad then
			for i = 1, #p._grad._storage do
				p._grad._storage[i] = math.clamp(p._grad._storage[i], -clip, clip)
			end
		end
	end
	task.synchronize()
end
function GC.clipNorm(params, maxNorm)
	task.desynchronize()
	local total = 0
	for _, p in params do
		if p._grad then
			for i = 1, #p._grad._storage do
				total += p._grad._storage[i] ^ 2
			end
		end
	end
	total = math.sqrt(total)
	if total > maxNorm then
		local scale = maxNorm / (total + 1e-9)
		for _, p in params do
			if p._grad then
				for i = 1, #p._grad._storage do
					p._grad._storage[i] *= scale
				end
			end
		end
	end
	task.synchronize()
end
return GC
