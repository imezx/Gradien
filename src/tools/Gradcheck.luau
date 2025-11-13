--!optimize 2
local Tape = require("../autograd/Tape")

local Gradcheck = {}

function Gradcheck.check(fn: (...number) -> (number, number), inputs: { any }, eps: number?, tol: number?)
	eps = eps or 1e-4
	tol = tol or 1e-2
	local out, grad = fn(table.unpack(inputs))
	Tape.backwardFrom(out, grad)
	for idx, t in ipairs(inputs) do
		if type(t) == "table" and t._storage then
			for i = 1, math.min(#t._storage, 50) do
				local orig = t._storage[i]
				t._storage[i] = orig + eps
				local out1, _ = fn(table.unpack(inputs))
				t._storage[i] = orig - eps
				local out2, _ = fn(table.unpack(inputs))
				t._storage[i] = orig
				local num = (out1 - out2) / (2 * (eps :: number))
				local ana = t._grad and t._grad._storage[i] or 0
				local rel = math.abs(num - ana) / math.max(1e-6, math.abs(num) + math.abs(ana))
				if rel > tol then
					print(
						string.format(
							"Gradcheck fail on input %d idx %d: num=%.6g ana=%.6g rel=%.3g",
							idx,
							i,
							num,
							ana,
							rel
						)
					)
					return false
				end
			end
		end
	end
	return true
end

return Gradcheck
