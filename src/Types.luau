--!strict
export type DType = "f64" | "f32" | "i32" | "u8"
export type Shape = { number }
export type Storage = { number }

export type Tensor = {
	_storage: Storage,
	_shape: Shape,
	_dtype: DType,
	_requiresGrad: boolean,
	_size: number,
	_strides: { number }?,
	_parents: { Tensor }?,
	_backprop: ((grad: Tensor) -> ())?,
	_grad: Tensor?,
}

export type Module<TIn = Tensor, TOut = Tensor> = {
	forward: (self: Module<TIn, TOut>, x: TIn) -> TOut,
	parameters: (self: Module<TIn, TOut>) -> { Tensor },
}

export type OptimizerState = {
	lr: number,
	b1: number,
	b2: number,
	eps: number,
	t: number,
	m: { [Tensor]: { [number]: number } },
	v: { [Tensor]: { [number]: number } },
}
export type Optimizer = {
	step: (self: Optimizer) -> (),
	zeroGrad: (self: Optimizer) -> (),
	getState: (self: Optimizer) -> OptimizerState?,
	setState: (self: Optimizer, state: OptimizerState?) -> (),
	setLr: (self: Optimizer, lr: number) -> (),
}

export type LossFn = (pred: Tensor, target: Tensor) -> (number, Tensor)

export type Dataset = {
	size: (self: Dataset) -> number,
	at: (self: Dataset, i: number) -> (Tensor, Tensor),
	batch: (self: Dataset, indices: { number }) -> (Tensor, Tensor),
	slice: (self: Dataset, i: number, j: number) -> (Tensor, Tensor),
}

export type BatchIter = () -> ({ number }, { number }) | nil

export type Snapshot = { SnapshotTensor }
export type TrainerSnapshot = {
	model: Snapshot,
	optimizer: OptimizerState?,
	step: number?,
	epoch: number?,
	bestMetric: number?,
}

export type SnapshotTensor = {
	shape: { number },
	dtype: DType,
	data: { number },
}

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
	lrScheduler: ((number) -> number)?,
	callbacks: Callbacks?,
}

export type FitOptions = {
	epochs: number?,
	stepsPerEpoch: number?,
	onMetric: ((number, number, number) -> ())?, -- (epoch, step, metric)
}

export type Trainer = {
	step: number,
	epoch: number,
	model: Module,
	optimizer: Optimizer,
	loss: LossFn,
	metric: ((Tensor, Tensor) -> number)?,
	lrScheduler: ((number) -> number)?,
	reportEvery: number?,
	callbacks: { CallbackFn },
	bestMetric: number?,
}

export type Sample = {
	name: string,
	total: number,
	self: number,
	count: number,
	avg: number,
	selfAvg: number,
	min: number,
	max: number,
	last: number,
	percent: number,
}

export type Stat = {
	name: string,
	total: number,
	selfTime: number,
	count: number,
	min: number,
	max: number,
	last: number,
}

export type Marker = {
	name: string,
	start: number,
	child: number,
}

export type Callable = (...any) -> ...any

return nil
