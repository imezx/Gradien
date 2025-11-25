--!native
--!optimize 2
local Tape = require("./autograd/Tape")
local Types = require("./Types")
local Losses = require("./nn/Losses")
local Metrics = require("./Metrics")
local Util = require("./Util")

local Trainer = {}
Trainer.__index = Trainer

local assert = Util.Assert

local function asArrayCallbacks(raw: Types.Callbacks?, default: { Types.CallbackFn }?): { Types.CallbackFn }
	if type(raw) ~= "table" then
		return default or {}
	end
	if not next(raw) then
		return raw :: { Types.CallbackFn }
	end

	local tbl = raw :: Types.CallbackTable
	local out: { Types.CallbackFn } = {}
	if tbl.onStep then
		local fn = tbl.onStep
		table.insert(out, function(event: Types.TrainerEvent, ctx: Types.CallbackContext)
			if event == "step_end" then
				fn(ctx)
			end
		end)
	end
	if tbl.onEpochEnd then
		local fn = tbl.onEpochEnd
		table.insert(out, function(event: Types.TrainerEvent, ctx: Types.CallbackContext)
			if event == "epoch_end" then
				fn(ctx)
			end
		end)
	end
	if tbl.onBest then
		local fn = tbl.onBest
		table.insert(out, function(event: Types.TrainerEvent, ctx: Types.CallbackContext)
			if event == "best" then
				fn(ctx)
			end
		end)
	end
	return out
end

local function fireCallbacks(callbacks: { Types.CallbackFn }, event: Types.TrainerEvent, ctx: Types.CallbackContext)
	for _, cb in callbacks do
		cb(event, ctx)
	end
end

local function runSimple(cfg: Types.SimpleConfig): ()
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
				if #callbacks > 0 then
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

		if #callbacks > 0 then
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
function Trainer.new(cfg: Types.TrainerConfig)
	local model = cfg.model
	local params = model:parameters()

	local opt: Types.Optimizer
	if cfg.optimizerFactory then
		opt = cfg.optimizerFactory(params)
	else
		assert(cfg.optimizer, "Trainer.new: provide optimizerFactory OR optimizer instance")
		opt = cfg.optimizer :: Types.Optimizer
	end

	local self = setmetatable({
		model = model,
		optimizer = opt,
		loss = cfg.loss,
		metric = cfg.metric,
		reportEvery = cfg.reportEvery or 10,
		lrScheduler = cfg.lrScheduler,
		callbacks = asArrayCallbacks(cfg.callbacks, {}),
		bestMetric = nil,
		step = 0,
		epoch = 1,
	}, Trainer)

	return self
end

--@param makeLoader () -> () -> (Tensor?, Tensor?)
--@param opts { epochs: number?, stepsPerEpoch: number?, onMetric: ((number, number, number) -> ())? }
function Trainer:fit(makeLoader: () -> () -> (Types.Tensor?, Types.Tensor?), opts: Types.FitOptions?)
	local fitOpts = (opts or {}) :: Types.FitOptions
	local epochs = fitOpts.epochs or self.epoch or 1
	local stepsPerEpoch = fitOpts.stepsPerEpoch
	local onMetric = fitOpts.onMetric

	for epoch = 1, epochs do
		self.epoch = epoch
		local loader = makeLoader()
		local lastLoss: number? = nil
		local lastMetric: number? = nil
		local stepsThisEpoch = 0

		while true do
			local X, Y = loader()
			if not X then
				break
			end
			self.step += 1
			stepsThisEpoch += 1

			self.optimizer:zeroGrad()
			local out = self.model:forward(X)
			local lossValue, grad = self.loss(out, Y)
			Tape.backwardFrom(out, grad)
			self.optimizer:step()
			if self.lrScheduler then
				local newLr = self.lrScheduler(self.step)
				if type(newLr) == "number" then
					if type(self.optimizer.setLr) == "function" then
						self.optimizer:setLr(newLr)
					elseif type(self.optimizer.setLR) == "function" then
						self.optimizer:setLR(newLr)
					end
				end
			end

			lastLoss = lossValue
			local metricValue: number? = nil

			if self.metric and self.reportEvery > 0 and (self.step % self.reportEvery == 0) then
				local m = self.metric(out, Y)
				metricValue = m
				lastMetric = m

				if onMetric then
					onMetric(epoch, self.step, m)
				end
				if self.bestMetric == nil or m > (self.bestMetric :: number) then
					self.bestMetric = m
					fireCallbacks(self.callbacks, "best", {
						epoch = epoch,
						step = self.step,
						loss = lossValue,
						metric = m,
						model = self.model,
						optimizer = self.optimizer,
					})
				end
			end

			fireCallbacks(self.callbacks, "step_end", {
				epoch = epoch,
				step = self.step,
				loss = lossValue,
				metric = metricValue,
				model = self.model,
				optimizer = self.optimizer,
			})

			if stepsPerEpoch and stepsThisEpoch >= stepsPerEpoch then
				break
			end
		end

		fireCallbacks(self.callbacks, "epoch_end", {
			epoch = epoch,
			step = self.step,
			loss = lastLoss,
			metric = lastMetric,
			model = self.model,
			optimizer = self.optimizer,
		})
	end
end

--@param cfg Types.TrainerConfig
--@param opts { smoothing: number?, loss: Types.LossFn?, metric: ((Types.Tensor, any) -> number)? }?
function Trainer.newClassification(
	cfg: Types.TrainerConfig,
	opts: {
		smoothing: number?,
		loss: Types.LossFn?,
		metric: ((Types.Tensor, any) -> number)?,
	}?
)
	local cfgOpts = (opts or {}) :: {
		smoothing: number?,
		loss: Types.LossFn?,
		metric: ((Types.Tensor, any) -> number)?,
	}
	local smoothing = cfgOpts.smoothing
	local lossFn = cfgOpts.loss or cfg.loss
	if not lossFn then
		lossFn = function(logits, target)
			return Losses.cross_entropy_backward(logits, target, smoothing)
		end
	end
	local metricFn = cfgOpts.metric or cfg.metric
	if not metricFn then
		metricFn = function(logits, target)
			return Metrics.accuracy(logits, target)
		end
	end
	return Trainer.new({
		model = cfg.model,
		optimizerFactory = cfg.optimizerFactory,
		optimizer = cfg.optimizer,
		loss = lossFn,
		metric = metricFn,
		reportEvery = cfg.reportEvery,
		lrScheduler = cfg.lrScheduler,
		callbacks = cfg.callbacks,
	})
end

function Trainer:run(loader: () -> (Types.Tensor?, Types.Tensor?), opts: Types.FitOptions?)
	local function makeLoader()
		return loader
	end
	self:fit(makeLoader, opts)
end

setmetatable(Trainer, {
	__call = function(_, cfg: Types.SimpleConfig)
		return runSimple(cfg)
	end,
})

return Trainer
