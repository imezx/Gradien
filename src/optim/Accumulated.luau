--!native
--!optimize 2
local Util = require("../Util")
local Types = require("../Types")
local assert = Util.Assert

local function scaleGrads(params: { Types.Tensor }, factor: number)
	task.desynchronize()
	for _, p in params do
		local g = p._grad
		if g then
			local gs = g._storage
			for i = 1, #gs do
				gs[i] *= factor
			end
		end
	end
	task.synchronize()
end

return function(opt: Types.Optimizer, steps: number, params: { Types.Tensor }?, normalize: boolean?)
	assert(steps >= 1, "Accumulated: steps must be >= 1")
	normalize = normalize == nil and true or normalize
	local counter: number = 0
	local invSteps: number = 1 / steps
	local doScale: boolean = normalize and params ~= nil
	return {
		step = function(_)
			counter += 1
			if counter < steps then
				return
			end
			if doScale then
				scaleGrads(params :: any, invSteps)
			end
			opt:step()
			opt:zeroGrad()
			counter = 0
		end,
		flush = function(_)
			if counter == 0 then
				return
			end
			if doScale then
				scaleGrads(params :: any, 1 / counter)
			end
			opt:step()
			opt:zeroGrad()
			counter = 0
		end,
		zeroGrad = function(_)
			counter = 0
			opt:zeroGrad()
		end,
		setLr = function(_, newLr: number)
			if opt.setLr then
				opt:setLr(newLr)
			end
		end,
		getState = function(_)
			local innerState = opt.getState and opt:getState() or nil
			return {
				steps = steps,
				counter = counter,
				normalize = normalize,
				optState = innerState,
			}
		end,
		setState = function(
			_,
			state: {
				steps: number,
				counter: number,
				normalize: boolean,
				optState: Types.OptimizerState,
			}
		)
			if state.steps ~= nil then
				steps = state.steps
				invSteps = 1 / steps
			end
			if state.counter ~= nil then
				counter = state.counter
			end
			if state.normalize ~= nil then
				normalize = state.normalize
				doScale = normalize and params ~= nil
			end
			if state.optState ~= nil and opt.setState then
				opt:setState(state.optState)
			end
		end,
	}
end
