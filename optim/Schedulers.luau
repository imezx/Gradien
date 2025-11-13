--!optimize 2
local S = {}

function S.step(lr: number, gamma: number, stepSize: number)
	return function(step: number)
		return lr * (gamma ^ math.floor(step / stepSize))
	end
end

function S.cosine(lr: number, lrMin: number, T: number)
	return function(step: number)
		local t = math.min(step / T, 1)
		return lrMin + 0.5 * (lr - lrMin) * (1 + math.cos(math.pi * t))
	end
end

function S.warmupCosine(lr: number, warmup: number, T: number)
	local base = S.cosine(lr, 0, T)
	return function(step: number)
		if step < warmup then
			return lr * (step / warmup)
		end
		return base(step - warmup)
	end
end

function S.oneCycle(lrMax: number, lrMin: number, T_up: number, T_total: number)
	return function(step: number)
		if step < T_up then
			return lrMax * (step / math.max(1, T_up))
		end
		local t = math.min(math.max(step - T_up, 0) / math.max(1, T_total - T_up), 1)
		return lrMin + 0.5 * (lrMax - lrMin) * (1 + math.cos(math.pi * t))
	end
end

function S.cosineRestarts(lr: number, T0: number, Tmult: number)
	local Tcur = T0
	local base = 0
	return function(step: number)
		local t = step - base
		task.desynchronize()
		while t >= Tcur do
			t -= Tcur
			base += Tcur
			Tcur *= (Tmult > 0 and Tmult or 2)
		end
		task.synchronize()
		local frac = t / math.max(1, math.floor(Tcur))
		return 0.5 * lr * (1 + math.cos(math.pi * frac))
	end
end

return S
