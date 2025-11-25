--!native
--!optimize 2
local function update(self, x: number)
	self.n += 1
	local d = x - self.mean
	self.mean += d / self.n
	self.m2 += d * (x - self.mean)
end

local function var(self): number
	if self.n < 2 then
		return 1.0
	end
	return self.m2 / (self.n - 1)
end

local function std(self): number
	return math.sqrt(var(self)) + self.eps
end

local function normalize(self, x: number): number
	return (x - self.mean) / std(self)
end

local function new(eps: number?)
	return {
		n = 0,
		mean = 0.0,
		m2 = 0.0,
		eps = eps or 1e-8,
		update = update,
		var = var,
		std = std,
		normalize = normalize,
	}
end

return new
