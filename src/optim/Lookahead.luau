--!optimize 2

--@param params { Tensor }
--@param base any -- base optimizer already constructed on params
--@param k integer? -- steps between sync (default 5)
--@param alpha number? -- interpolation factor (default 0.5)
return function(params, base, k: number?, alpha: number?)
	k = k or 5; alpha = alpha or 0.5
	local step = 0
	local slow = {}
	for i = 1, #params do
		local t = params[i]
		local buf = table.create(#t._storage, 0.0)
		for j = 1, #t._storage do buf[j] = t._storage[j] end
		slow[i] = buf
	end

	local opt = {}
	opt.base = base
	function opt:zeroGrad()
		base:zeroGrad()
	end
	function opt:step()
		base:step()
		step += 1
		if step % k == 0 then
			task.desynchronize()
			for i = 1, #params do
				local p = params[i]
				local s = slow[i]
				for j = 1, #p._storage do
					s[j] = s[j] + alpha * (p._storage[j] - s[j])
					p._storage[j] = s[j]
				end
			end
			task.synchronize()
		end
	end
	return opt
end
