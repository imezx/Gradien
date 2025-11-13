--!optimize 2
local Tape = require("./autograd/Tape")
local Types = require("./Types")

type Tensor = Types.Tensor
type Module = Types.Module
type Optimizer = Types.Optimizer
type LossFn = Types.LossFn

export type TrainerEvent = "step_end" | "epoch_end" | "best"

export type CallbackContext = {
	epoch: number,
	step: number?,
	loss: number?,
	metric: number?,
	model: Module,
	optimizer: Optimizer,
}

export type CallbackFn = (TrainerEvent, CallbackContext) -> ()

export type CallbackTable = {
	onStep: ((CallbackContext) -> ())?,
	onEpochEnd: ((CallbackContext) -> ())?,
	onBest: ((CallbackContext) -> ())?,
}

export type Callbacks = { CallbackFn } | CallbackTable

export type SimpleConfig = {
	model: Module,
	optimizer: (params: { Tensor }) -> Optimizer,
	loss: LossFn,
	loader: () -> (Tensor, Tensor),
	metric: ((Tensor, Tensor) -> number)?,
	steps: number?,
	reportEvery: number?,
	onMetric: ((number) -> ())?,
	callbacks: Callbacks?,
}

export type TrainerConfig = {
	model: Module,
	optimizerFactory: (({ Tensor }) -> Optimizer)?,
	optimizer: Optimizer?,
	loss: LossFn,
	metric: ((Tensor, Tensor) -> number)?,
	reportEvery: number?,
	callbacks: Callbacks?,
}

export type FitOptions = {
	epochs: number?,
	stepsPerEpoch: number?,
	onMetric: ((number, number, number) -> ())?, -- (epoch, step, metric)
}

local Trainer = {}
Trainer.__index = Trainer

local function asArrayCallbacks(raw: Callbacks?, default: { CallbackFn }?): { CallbackFn }
	if type(raw) ~= "table" then
		return default or {}
	end
	if next(raw) then
		return raw :: { CallbackFn }
	end

	local tbl = raw :: CallbackTable
	local out: { CallbackFn } = {}
	if tbl.onStep then
		local fn = tbl.onStep
		table.insert(out, function(event: TrainerEvent, ctx: CallbackContext)
			if event == "step_end" then
				fn(ctx)
			end
		end)
	end
	if tbl.onEpochEnd then
		local fn = tbl.onEpochEnd
		table.insert(out, function(event: TrainerEvent, ctx: CallbackContext)
			if event == "epoch_end" then
				fn(ctx)
			end
		end)
	end
	if tbl.onBest then
		local fn = tbl.onBest
		table.insert(out, function(event: TrainerEvent, ctx: CallbackContext)
			if event == "best" then
				fn(ctx)
			end
		end)
	end
	return out
end

local function fireCallbacks(callbacks: { CallbackFn }, event: TrainerEvent, ctx: CallbackContext)
	for _, cb in callbacks do
		cb(event, ctx)
	end
end

local function runSimple(cfg: SimpleConfig): ()
	task.desynchronize()
	local model = cfg.model
	local opt = cfg.optimizer(model:parameters())
	local lossFn = cfg.loss
	local loader = cfg.loader
	local metricFn = cfg.metric
	local steps = cfg.steps or 100
	local reportEvery = cfg.reportEvery or 10
	local onMetric = cfg.onMetric
	local callbacks = asArrayCallbacks(cfg.callbacks, nil)

	local bestMetric: number? = nil

	for step = 1, steps do
		opt:zeroGrad()
		local X, Y = loader()
		local out = model:forward(X)
		local lossValue, grad = lossFn(out, Y)
		Tape.backwardFrom(out, grad)
		opt:step()

		local metricValue: number? = nil
		if metricFn and (step % reportEvery == 0) then
			local m = metricFn(out, Y)
			metricValue = m
			if onMetric then
				onMetric(m)
			end
			if bestMetric == nil or m > bestMetric then
				bestMetric = m
				if next(callbacks) then
					fireCallbacks(callbacks, "best", {
						epoch = 1,
						step = step,
						loss = lossValue,
						metric = m,
						model = model,
						optimizer = opt,
					})
				end
			end
		end

		if next(callbacks) then
			fireCallbacks(callbacks, "step_end", {
				epoch = 1,
				step = step,
				loss = lossValue,
				metric = metricValue,
				model = model,
				optimizer = opt,
			})
		end
	end
	task.synchronize()
end

--@param cfg { model: Module, optimizerFactory: (({ Tensor }) -> Optimizer)?, optimizer: Optimizer?, loss: LossFn, metric: ((Tensor, Tensor) -> number)?, reportEvery: number?, callbacks: Callbacks? }
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
		callbacks = asArrayCallbacks(cfg.callbacks, {}),
		bestMetric = nil,
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
		local lastLoss: number? = nil
		local lastMetric: number? = nil

		while true do
			local X, Y = loader()
			if not X then
				break
			end
			step += 1

			self.optimizer:zeroGrad()
			local out = self.model:forward(X)
			local lossValue, grad = self.loss(out, Y)
			Tape.backwardFrom(out, grad)
			self.optimizer:step()

			lastLoss = lossValue
			local metricValue: number? = nil

			if self.metric and self.reportEvery > 0 and (step % self.reportEvery == 0) then
				local m = self.metric(out, Y)
				metricValue = m
				lastMetric = m

				if onMetric then
					onMetric(epoch, step, m)
				end
				if self.bestMetric == nil or m > (self.bestMetric :: number) then
					self.bestMetric = m
					fireCallbacks(self.callbacks, "best", {
						epoch = epoch,
						step = step,
						loss = lossValue,
						metric = m,
						model = self.model,
						optimizer = self.optimizer,
					})
				end
			end

			fireCallbacks(self.callbacks, "step_end", {
				epoch = epoch,
				step = step,
				loss = lossValue,
				metric = metricValue,
				model = self.model,
				optimizer = self.optimizer,
			})

			if stepsPerEpoch and step >= stepsPerEpoch then
				break
			end
		end

		fireCallbacks(self.callbacks, "epoch_end", {
			epoch = epoch,
			step = step,
			loss = lastLoss,
			metric = lastMetric,
			model = self.model,
			optimizer = self.optimizer,
		})
	end
end

function Trainer:run(loader: () -> (Tensor?, Tensor?), opts: FitOptions?)
	local function makeLoader()
		return loader
	end
	self:fit(makeLoader, opts)
end

setmetatable(Trainer, {
	__call = function(_, cfg: SimpleConfig)
		return runSimple(cfg)
	end,
})

return Trainer
