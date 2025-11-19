# Autograd (Tape) <Badge type="tip" text="Core" />

The `Tape` module records operations and manages the backward pass.

## Functions

### `.backwardFrom` <Badge type="info" text="Parallel" />
Computes the gradient of current tensor w.r.t. graph leaves.

::: code-group
```lua [Definition]
(y: Tensor, grad: Tensor) -> ()
```
```lua [Example]
Tape.backwardFrom(loss, Tensor.ones(loss._shape))
```
:::

### `.grad`
Computes and returns the sum of gradients of outputs with respect to the inputs.

::: code-group
```lua [Definition]
(f: (...any) -> Tensor, inputs: {Tensor}) -> {Tensor?}
```
:::

### `.noGrad`
Disables gradient tracking for the duration of the function `fn`. Useful for inference.

::: code-group
```lua [Definition]
(fn: () -> ()) -> ()
```
```lua [Example]
Tape.noGrad(function()
    -- Operations here won't be recorded
    local pred = model:forward(input)
end)
```
:::

### `.makeNode`
Internal function used to register a new operation in the computation graph.

::: code-group
```lua [Definition]
(out: Tensor, parents: {Tensor}, back: (grad: Tensor) -> ()) -> Tensor
```
:::