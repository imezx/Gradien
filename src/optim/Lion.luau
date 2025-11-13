--!optimize 2
-- Chen et al., 2023. Update uses sign of momentum (no per-param v).

--@param params {any}
--@param lr number
--@param beta1 number? @default 0.9
--@param beta2 number? @default 0.99
--@param weightDecay number? @default 0.0 (decoupled)
return function(params, lr: number, beta1: number?, beta2: number?, weightDecay: number?)
	local state = { m = {} }
	beta1, beta2 = beta1 or 0.9, beta2 or 0.99
	weightDecay = weightDecay or 0.0
	for i = 1, #params do
		local t = params[i]
		local m = table.create(#t._storage, 0.0)
		state.m[i] = m
	end
	local opt = {}
	opt.lr = lr
	function opt:zeroGrad()
		for i = 1, #params do
			local g = params[i]._grad
			if g then task.desynchronize(); for k = 1, #g._storage do g._storage[k] = 0 end; task.synchronize() end
		end
	end
	function opt:step()
		task.desynchronize()
		for i = 1, #params do
			local p = params[i]
			local g = p._grad
			if g then
				local mi = state.m[i]
				for k = 1, #p._storage do
					if weightDecay ~= 0 then
						p._storage[k] -= opt.lr * weightDecay * p._storage[k]
					end
					-- momentum
					mi[k] = beta1 * mi[k] + (1 - beta1) * g._storage[k]
					-- lookahead step
					local update = math.sign(mi[k]) * opt.lr
					p._storage[k] -= update
					-- second momentum blend
					mi[k] = beta2 * mi[k] + (1 - beta2) * g._storage[k]
				end
			end
		end
		task.synchronize()
	end
	return opt
end
