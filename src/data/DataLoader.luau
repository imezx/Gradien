--!optimize 2
local Types = require("../Types")

return function(
	dataset: Types.Dataset,
	batchSize: number,
	shuffle: boolean,
	generator: { randint: (a: number, b: number) -> number }?,
	drop_last: boolean?
)
	local N = dataset:size()
	generator = generator or {
		randint = function(a: number, b: number)
			return math.random(a, b)
		end,
	}
	local order = table.create(N, 0)
	for i = 1, N do
		order[i] = i
	end
	if shuffle then
		for i = N, 2, -1 do
			local j = generator.randint(1, i)
			order[i], order[j] = order[j], order[i]
		end
	end
	local idx = 1
	return function()
		if idx > N or (drop_last and (N - idx + 1) < batchSize) then
			return nil
		end
		local last = math.min(idx + batchSize - 1, N)
		local count = last - idx + 1
		local batchIdx = table.create(count)
		for i = 1, count do
			batchIdx[i] = order[idx + i - 1]
		end
		idx = last + 1
		if dataset.batch then
			return dataset:batch(batchIdx)
		elseif dataset.slice and not shuffle then
			return dataset:slice(batchIdx[1], batchIdx[#batchIdx])
		end
		return dataset:at(batchIdx[1])
	end
end
