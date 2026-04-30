--!native
--!optimize 2
local S = {}

--@param lr number
--@param gamma number
--@param stepSize number
-- decays lr by gamma every stepSize steps
function S.step(lr: number, gamma: number, stepSize: number): (step: number) -> number
	return function(step: number)
		return lr * (gamma ^ math.floor(step / stepSize))
	end
end

--@param lr number
--@param gamma number
-- lr * gamma^step
function S.exponential(lr: number, gamma: number): (step: number) -> number
	return function(step: number)
		return lr * (gamma ^ step)
	end
end

--@param lr number
--@param lrMin number
--@param T number
--@param power number?
-- polynomial decay from lr -> lrMin over T steps
function S.polynomial(lr: number, lrMin: number, T: number, power: number?): (step: number) -> number
	local p = power or 1.0
	return function(step: number)
		local t = math.min(step / math.max(T, 1), 1)
		return (lr - lrMin) * ((1 - t) ^ p) + lrMin
	end
end

--@param lr number
-- constant lr
function S.constant(lr: number): () -> number
	return function()
		return lr
	end
end

--@param lr number
--@param lrMin number
--@param T number
-- cosine annealing from lr -> lrMin over T steps
function S.cosine(lr: number, lrMin: number, T: number): (step: number) -> number
	return function(step: number)
		local t = math.min(step / T, 1)
		return lrMin + 0.5 * (lr - lrMin) * (1 + math.cos(math.pi * t))
	end
end

--@param lr number
--@param warmup number
--@param T number
--@param lrMin number?
-- linear warmup then cosine decay. lrMin defaults to 0
function S.warmupCosine(lr: number, warmup: number, T: number, lrMin: number?): (step: number) -> number
	local min = lrMin or 0
	local decay = S.cosine(lr, min, T)
	return function(step: number)
		if step < warmup then
			return lr * (step / math.max(warmup, 1))
		end
		return decay(step - warmup)
	end
end

--@param lr number
--@param warmup number
--@param T number
--@param lrMin number?
-- linear warmup then linear decay to lrMin
function S.warmupLinear(lr: number, warmup: number, T: number, lrMin: number?): (step: number) -> number
	local min = lrMin or 0
	return function(step: number)
		if step < warmup then
			return lr * (step / math.max(warmup, 1))
		end
		local t = math.min((step - warmup) / math.max(T - warmup, 1), 1)
		return lr + (min - lr) * t
	end
end

--@param lr number
--@param lrMin number
--@param warmup number
--@param T number
--@param power number?
-- linear warmup then polynomial decay
function S.warmupPolynomial(lr: number, lrMin: number, warmup: number, T: number, power: number?): (step: number) -> number
	local decay = S.polynomial(lr, lrMin, T - warmup, power)
	return function(step: number)
		if step < warmup then
			return lr * (step / math.max(warmup, 1))
		end
		return decay(step - warmup)
	end
end

--@param lrMax number
--@param lrMin number
--@param T_up number
--@param T_total number
-- linear warmup to lrMax, then cosine down to lrMin
function S.oneCycle(lrMax: number, lrMin: number, T_up: number, T_total: number): (step: number) -> number
	return function(step: number)
		if step < T_up then
			return lrMax * (step / math.max(1, T_up))
		end
		local t = math.min(math.max(step - T_up, 0) / math.max(1, T_total - T_up), 1)
		return lrMin + 0.5 * (lrMax - lrMin) * (1 + math.cos(math.pi * t))
	end
end

--@param lrMax number
--@param T0 number
--@param Tmult number
-- cosine annealing with warm restarts
function S.cosineRestarts(lr: number, T0: number, Tmult: number): (step: number) -> number
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

--@param lr number
--@param factor number?
--@param patience number?
--@param threshold number?
--@param mode string?
function S.reduceOnPlateau(lr: number, factor: number?, patience: number?, threshold: number?, mode: string?)
	local f = factor or 0.1
	local p = patience or 10
	local th = threshold or 1e-4
	local m = mode or "min" -- "min" or "max"

	local currentLr = lr
	local best = (m == "min") and math.huge or -math.huge
	local wait = 0

	local function isBetter(val: number): boolean
		if m == "min" then
			return val < best - th
		end
		return val > best + th
	end

	return {
		update = function(_, metric: number)
			if isBetter(metric) then
				best = metric
				wait = 0
			else
				wait += 1
				if wait >= p then
					currentLr = math.max(currentLr * f, 1e-8)
					wait = 0
				end
			end
		end,
		getLr = function(_): number
			return currentLr
		end,
		reset = function(_)
			currentLr = lr
			best = (m == "min") and math.huge or -math.huge
			wait = 0
		end,
	}
end

S.linearWarmupThenCosine = S.warmupCosine

return S
