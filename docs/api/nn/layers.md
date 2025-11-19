# Layers <Badge type="tip" text="Module" />

Layers are the building blocks of neural networks. They store learnable parameters (`weights` and `biases`).

## `NN.Linear`

Applies a linear transformation to the incoming data: `y = x * W^T + b`.

::: code-group
```lua [Definition]
(
    inFeatures: number,
    outFeatures: number,
    initializer: ((fanIn, fanOut) -> number)?
) -> Module
```
```lua [Example]
local layer = Gradien.NN.Linear(128, 64)
```
:::

## `NN.Sequential`

A container that chains modules together. Data flows through them in the order they are defined.

::: code-group
```lua [Definition]
(layers: { Module | (Tensor)->Tensor }) -> Module
```
```lua [Example]
local model = Gradien.NN.Sequential({
    Gradien.NN.Linear(10, 20),
    Gradien.NN.Linear(20, 5)
})
```
:::