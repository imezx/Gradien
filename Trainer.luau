--!optimize 2
local Tape = require("./autograd/Tape")
local Types = require("./Types")
return function(cfg: {
	model: Types.Module,
	optimizer: (params: { Types.Tensor }) -> Types.Optimizer,
	loss: Types.LossFn,
	loader: () -> (Types.Tensor, Types.Tensor),
	metric: (Types.Tensor, Types.Tensor) -> number,
	steps: number?,
	reportEvery: number?,
	onMetric: (number) -> (),
}): ()
	task.desynchronize()
	local model = cfg.model
	local opt = cfg.optimizer(model:parameters())
	local lossFn = cfg.loss
	local loader = cfg.loader
	local metric = cfg.metric
	local steps = cfg.steps or 1000
	local reportEvery = cfg.reportEvery or 50
	local onMetric = cfg.onMetric
	for step = 1, steps do
		opt:zeroGrad()
		local X, Y = loader()
		local out = model:forward(X)
		local _, grad = lossFn(out, Y)
		Tape.backwardFrom(out, grad)
		opt:step()
		if metric and step % reportEvery == 0 and onMetric then
			onMetric(metric(out, Y))
		end
	end
	task.synchronize()
end
