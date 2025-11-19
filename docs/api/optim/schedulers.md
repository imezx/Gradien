# Learning Rate Schedulers <Badge type="info" text="Utils" />

Located in `Gradien.Optim.Schedulers`. These functions return a callback `(step) -> newLr`.

## Standard

### `step`
Decays LR by `gamma` every `stepSize`.
```lua
local sched = Schedulers.step(lr, 0.9, 100)
```

### `cosine`
Cosine annealing from `lr` down to `lrMin` over `T` steps.
```lua
local sched = Schedulers.cosine(1e-3, 1e-6, 1000)
```

### `cosineRestarts` <Badge type="info" text="Parallel" />
Cosine annealing with warm restarts. `T0` is initial period, `Tmult` scales period after restart.
```lua
local sched = Schedulers.cosineRestarts(1e-3, 100, 2.0)
```

## Advanced

### `warmupCosine`
Linearly warms up from 0 to `lr`, then decays via cosine.
```lua
local sched = Schedulers.warmupCosine(1e-3, 100, 1000)
```

### `oneCycle`
1-Cycle policy. rapid increase to `lrMax` then decay.
```lua
local sched = Schedulers.oneCycle(maxLr, minLr, warmupSteps, totalSteps)
```