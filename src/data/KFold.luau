--!native
--!optimize 2
local Types = require("../Types")
return function(N: number, K: number): Types.BatchIter
	local order: { number } = {}
	for i = 1, N do
		order[i] = i
	end
	for i = N, 2, -1 do
		local j = math.random(i)
		order[i], order[j] = order[j], order[i]
	end
	local fold = 1
	return function()
		if fold > K then
			return nil
		end
		local size = math.floor(N / K)
		local s = (fold - 1) * size + 1
		local e = (fold == K) and N or (s + size - 1)
		local test: { number } = {}
		for i = s, e do
			table.insert(test, order[i])
		end
		local train: { number } = {}
		for i = 1, N do
			if i < s or i > e then
				table.insert(train, order[i])
			end
		end
		fold += 1
		return train, test
	end
end
