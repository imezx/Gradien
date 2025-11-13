--!strict
export type DType = "f64" | "f32" | "i32" | "u8" | string
export type Shape = { number }
export type Storage = { number }

export type Tensor = {
	_storage: Storage,
	_shape: Shape,
	_dtype: DType,
	_requiresGrad: boolean,
	_parents: { Tensor }?,
	_backprop: ((grad: Tensor) -> ())?,
	_grad: Tensor?,
}

export type Module<TIn = Tensor, TOut = Tensor> = {
	forward: (self: Module<TIn, TOut>, x: TIn) -> TOut,
	parameters: (self: Module<TIn, TOut>) -> { Tensor },
}

export type Optimizer = {
	step: (self: Optimizer) -> (),
	zeroGrad: (self: Optimizer) -> (),
}

export type LossFn = (pred: Tensor, target: Tensor) -> (number, Tensor)

export type Dataset = {
	size: (self: Dataset) -> number,
	at: (self: Dataset, i: number) -> (Tensor, Tensor),
	batch: (self: Dataset, indices: { number }) -> (Tensor, Tensor),
	slice: (self: Dataset, i: number, j: number) -> (Tensor, Tensor),
}

export type BatchIter = () -> ({ number }, { number }) | nil

return nil
