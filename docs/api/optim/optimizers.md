# Optimizers <Badge type="tip" text="Optimization" />

Gradien includes a wide variety of optimizers for different use cases.

## Adaptive Methods

### `Adam`
Adaptive Moment Estimation. Good default.
::: code-group
```lua [Constructor]
(params: {Tensor}, lr: number, b1: num?, b2: num?, eps: num?) -> Optimizer
```
:::

### `AdamW` <Badge type="tip" text="Modern" />
Adam with decoupled weight decay. Often generalizes better than Adam.
::: code-group
```lua [Constructor]
(
    params: {Tensor}, 
    lr: number, 
    wd: number, -- Weight Decay
    b1: number?, 
    b2: number?, 
    eps: number?
) -> Optimizer
```
:::

### `Lion` <Badge type="tip" text="New" />
Evolved Sign Momentum. Memory efficient and often faster than Adam.
::: code-group
```lua [Constructor]
(
    params: {Tensor}, 
    lr: number, 
    beta1: number?, -- default 0.9
    beta2: number?, -- default 0.99
    weightDecay: number?
) -> Optimizer
```
:::

### `Adafactor`
Memory-efficient adaptive optimization. Can operate in 2D factored mode to save memory on large matrices.
::: code-group
```lua [Constructor]
(
    params: {Tensor},
    lr: number,
    beta2: number?,
    eps: number?,
    clipThreshold: number?,
    weightDecay: number?
) -> Optimizer
```
:::

### `RMSProp`
Maintains a moving average of the squared gradient.
::: code-group
```lua [Constructor]
(params: {Tensor}, lr: number, decay: number?, eps: number?) -> Optimizer
```
:::

### `Adagrad`
Adapts learning rates based on the frequency of parameters updates.
::: code-group
```lua [Constructor]
(params: {Tensor}, lr: number, eps: number?) -> Optimizer
```
:::

## Stochastic Methods

### `SGD`
Stochastic Gradient Descent with Momentum and Nesterov acceleration.
::: code-group
```lua [Constructor]
(params: {Tensor}, lr: number, momentum: number?, nesterov: boolean?) -> Optimizer
```
:::