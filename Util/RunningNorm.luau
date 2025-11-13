--!optimize 2
local RN = {}
RN.__index = RN

function RN.new(eps: number?)
	return setmetatable({ n = 0, mean = 0.0, m2 = 0.0, eps = eps or 1e-8 }, RN)
end

function RN:update(x: number)
	self.n += 1
	local d = x - self.mean
	self.mean += d / self.n
	self.m2 += d * (x - self.mean)
end

function RN:var(): number
	if self.n < 2 then
		return 1.0
	end
	return self.m2 / (self.n - 1)
end

function RN:std(): number
	return math.sqrt(self:var()) + self.eps
end

function RN:normalize(x: number): number
	return (x - self.mean) / self:std()
end

return RN
