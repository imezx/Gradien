--!optimize 2
local NStep = {}
function NStep.returns(rewards: { number }, dones: { boolean }, gamma: number, n: number): { number }
	task.desynchronize()
	local out = table.create(#rewards, 0)
	for t = 1, #rewards do
		local G, g = 0, 1
		for k = 0, n - 1 do
			local i = t + k
			if i > #rewards or dones[i] then
				G += g * (rewards[i] or 0)
				break
			end
			G += g * rewards[i]
			g *= gamma
		end
		out[t] = G
	end
	task.synchronize()
	return out
end
return NStep
