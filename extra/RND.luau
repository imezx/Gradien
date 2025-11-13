--!optimize 2
local Tape = require("../autograd/Tape")
local Tensor = require("../Tensor")
local Types = require("../Types")

local RND = {}
RND.__index = RND

--@param target any   -- module with :forward({D,B})->{E,B}; randomly init, never trained
--@param predictor any -- trainable module with same input/output dims as target
--@param optimizer any -- optimizer over predictor:parameters()
--@param norm? table  -- optional running norm with :update(x) and :normalize(x)
function RND.new(
	target: Types.Module<Types.Tensor, Types.Tensor>,
	predictor: Types.Module<Types.Tensor, Types.Tensor>,
	optimizer: Types.Optimizer,
	norm: {
		update: (self: { update: (x: number) -> (), normalize: (x: number) -> number }, x: number) -> number,
		normalize: (self: { update: (x: number) -> number, normalize: (x: number) -> number }, x: number) -> number,
	}?
)
	local self = setmetatable({}, RND)
	self.target = target
	self.predictor = predictor
	self.opt = optimizer
	self.norm = norm
	return self
end

--@param s Tensor
--@return number -- intrinsic reward (normalized if norm given)
function RND:step(s: Types.Tensor): number
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

return RND
