--!native
--!optimize 2
local RNG = {}
local _rng = Random.new()

function RNG.seed(s: number)
	_rng = Random.new(math.floor(s))
	return s
end

function RNG.uniform(a: number?, b: number?, rng: Random?)
	a = a or 0
	b = b or 1
	return (rng or _rng):NextNumber(a, b)
end

function RNG.randint(lo: number, hi: number, rng: Random?)
	-- inclusive
	return lo + math.floor((rng or _rng):NextNumber(0, 1) * (hi - lo + 1))
end

function RNG.generator()
	return {
		randint = function(_, lo: number, hi: number, rng: Random?): number
			return RNG.randint(lo, hi, rng)
		end,
		random = function(_, rng: Random?): number
			return RNG.uniform(0, 1, rng)
		end,
		uniform = function(_, a: number?, b: number?, rng: Random?): number
			return RNG.uniform(a, b, rng)
		end,
	}
end

return RNG
