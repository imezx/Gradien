# Optimization Wrappers <Badge type="warning" text="Advanced" />

These modules wrap an existing optimizer (the "base") to add functionality like averaging or accumulation.

## `Lookahead`
Improves stability by keeping a set of "slow weights" that interpolate with the "fast weights" trained by the base optimizer.

::: code-group
```lua [Constructor]
(
    params: {Tensor},
    base: Optimizer, -- Existing optimizer instance
    k: number?,      -- Sync every k steps (default 5)
    alpha: number?   -- Interpolation factor (default 0.5)
) -> Optimizer
```
```lua [Example]
local base = Gradien.Optim.Adam(params, 1e-3)
local opt = Gradien.Optim.Lookahead(params, base, 5, 0.5)
```
:::

## `Accumulated`
Simulates a larger batch size by accumulating gradients over multiple steps before updating.

::: code-group
```lua [Constructor]
(
    opt: Optimizer,
    steps: number,      -- Steps to accumulate
    params: {Tensor}?,  -- Needed if normalize=true
    normalize: boolean? -- Average grads instead of sum
) -> Optimizer
```
:::

## `EMA` (Exponential Moving Average)
Maintains a shadow copy of model parameters that updates smoothly. Often used for inference or target networks in RL.

::: code-group
```lua [Constructor]
(params: {Tensor}, decay: number) -> EMA_Handler
```
```lua [Methods]
ema:update(params)  -- Call after opt:step()
ema:apply(params)   -- Swap weights to EMA for eval
ema:restore(params) -- Swap back to training weights
```
:::