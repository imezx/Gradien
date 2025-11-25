--!native
--!optimize 2

--@param N number -- total items
--@param testRatio number? -- fraction in test split (default 0.2)
--@param stratify {number}? -- optional class per item (1..C), length N
--@param rng Random.new? -- optional RNG for deterministic splits
--@return {number}, {number} -- trainIdx, testIdx (1-based indices into your data)
return function(N: number, testRatio: number?, stratify: { number }?, rng: Random?): ({ number }, { number })
	rng = rng or Random.new()
	testRatio = testRatio or 0.2
	local idx = table.create(N, 0)
	for i = 1, N do
		idx[i] = i
	end
	if stratify then
		local by = {}
		for i = 1, N do
			local c = stratify[i]
			if not by[c] then
				by[c] = {}
			end
			table.insert(by[c], i)
		end
		local train, test = {}, {}
		for _, arr in by do
			for i = #arr, 2, -1 do
				local j = rng:NextInteger(1, i)
				arr[i], arr[j] = arr[j], arr[i]
			end
			local cut = math.floor(#arr * (1 - testRatio :: number))
			for i = 1, #arr do
				if i <= cut then
					table.insert(train, arr[i])
				else
					table.insert(test, arr[i])
				end
			end
		end
		return train, test
	end
	for i = N, 2, -1 do
		local j = rng:NextInteger(1, i)
		idx[i], idx[j] = idx[j], idx[i]
	end
	local cut = math.floor(N * (1 - testRatio :: number))
	local train = {}
	local test = {}
	for i = 1, N do
		if i <= cut then
			table.insert(train, idx[i])
		else
			table.insert(test, idx[i])
		end
	end
	return train, test
end
