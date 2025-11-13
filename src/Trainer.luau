--!optimize 2
local Tape = require("./autograd/Tape")
local Types = require("./Types")

type Tensor = Types.Tensor
type Module = Types.Module
type Optimizer = Types.Optimizer
type LossFn = Types.LossFn
export type SimpleConfig = {
	model: Module,
	optimizer: (params: { Tensor }) -> Optimizer,
	loss: LossFn,
	loader: () -> (Tensor, Tensor),
	metric: ((Tensor, Tensor) -> number)?,
	steps: number?,
	reportEvery: number?,
	onMetric: ((number) -> ())?,
}
export type TrainerConfig = {
	model: Module,
	optimizerFactory: (({ Tensor }) -> Optimizer)?,
	optimizer: Optimizer?,
	loss: LossFn,
	metric: ((Tensor, Tensor) -> number)?,
	reportEvery: number?,
	callbacks: { (string, any) -> () }?,
}

export type FitOptions = {
	epochs: number?,
	stepsPerEpoch: number?,
	onMetric: ((number, number, number) -> ())?, -- (epoch, step, metric)
}

local Trainer = {}
Trainer.__index = Trainer

local function runSimple(cfg: SimpleConfig): ()
	task.desynchronize()
	local model = cfg.model
	local opt = cfg.optimizer(model:parameters())
	local lossFn = cfg.loss
	local loader = cfg.loader
	local metric = cfg.metric
	local steps = cfg.steps or 100
	local reportEvery = cfg.reportEvery or 10
	local onMetric = cfg.onMetric

	for step = 1, steps do
		opt:zeroGrad()
		local X, Y = loader()
		local out = model:forward(X)
		local _, grad = lossFn(out, Y)
		Tape.backwardFrom(out, grad)
		opt:step()

		if metric and onMetric and (step % reportEvery == 0) then
			onMetric(metric(out, Y))
		end
	end
	task.synchronize()
end

--@param cfg { model: Module, optimizerFactory: (({ Tensor }) -> Optimizer)?, optimizer: Optimizer?, loss: LossFn, metric: ((Tensor, Tensor) -> number)?, reportEvery: number?, callbacks: { (string, any) -> () }? }
--@return Trainer
function Trainer.new(cfg: TrainerConfig)
	local model = cfg.model
	local params = model:parameters()

	local opt: Optimizer
	if cfg.optimizerFactory then
		opt = cfg.optimizerFactory(params)
	else
		assert(cfg.optimizer, "Trainer.new: provide optimizerFactory OR optimizer instance")
		opt = cfg.optimizer :: Optimizer
	end

	local self = setmetatable({
		model = model,
		optimizer = opt,
		loss = cfg.loss,
		metric = cfg.metric,
		reportEvery = cfg.reportEvery or 10,
		callbacks = cfg.callbacks or {},
	}, Trainer)

	return self
end

--@param makeLoader () -> () -> (Tensor?, Tensor?)
--@param opts { epochs: number?, stepsPerEpoch: number?, onMetric: ((number, number, number) -> ())? }
function Trainer:fit(makeLoader: () -> () -> (Tensor?, Tensor?), opts: FitOptions?)
	opts = opts or {}
	local epochs = opts.epochs or 1
	local stepsPerEpoch = opts.stepsPerEpoch
	local onMetric = opts.onMetric

	for epoch = 1, epochs do
		local loader = makeLoader()
		local step = 0

		while true do
			local X, Y = loader()
			if not X then
				break
			end
			step += 1

			self.optimizer:zeroGrad()
			local out = self.model:forward(X)
			local _, grad = self.loss(out, Y)
			Tape.backwardFrom(out, grad)
			self.optimizer:step()

			if self.metric and self.reportEvery > 0 and (step % self.reportEvery == 0) then
				local m = self.metric(out, Y)
				if onMetric then
					onMetric(epoch, step, m)
				end
			end
			if stepsPerEpoch and step >= stepsPerEpoch then
				break
			end
		end

		for _, cb in self.callbacks do
			cb("epoch_end", {
				epoch = epoch,
				model = self.model,
			})
		end
	end
end

setmetatable(Trainer, {
	__call = function(_, cfg: SimpleConfig)
		return runSimple(cfg)
	end,
})

return Trainer
