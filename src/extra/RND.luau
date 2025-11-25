--!native
--!optimize 2
local Tape = require("../autograd/Tape")
local Tensor = require("../Tensor")
local Types = require("../Types")

local function step(self, s: Types.Tensor): number
	self.opt:zeroGrad()
	local g = self.target:forward(s) :: Types.Tensor -- {E,1}
	local f = self.predictor:forward(s) :: Types.Tensor -- {E,1}
	local loss = 0
	local grad = Tensor.zeros(f._shape)
	task.desynchronize()
	for i = 1, #f._storage do
		local d = f._storage[i] - g._storage[i]
		loss += d * d
		grad._storage[i] = 2 * d
	end
	task.synchronize()
	Tape.backwardFrom(f, grad)
	self.opt:step()
	local r = loss
	if self.norm then
		self.norm:update(r)
		r = self.norm:normalize(r)
	end
	return r
end

local function new(
	target: Types.Module<Types.Tensor, Types.Tensor>,
	predictor: Types.Module<Types.Tensor, Types.Tensor>,
	optimizer: Types.Optimizer,
	norm: {
		update: (self: { update: (x: number) -> (), normalize: (x: number) -> number }, x: number) -> number,
		normalize: (self: { update: (x: number) -> number, normalize: (x: number) -> number }, x: number) -> number,
	}?
)
	return {
		target = target,
		predictor = predictor,
		opt = optimizer,
		norm = norm,
		step = step,
	}
end

return { new = new }
