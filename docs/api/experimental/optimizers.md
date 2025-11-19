# Experimental Optimizers <Badge type="warning" text="Experimental" />

Located in `Gradien.Experimental.Optim`. These optimizers use population-based metaheuristics rather than (or in addition to) standard backpropagation.

## `SwarmPSO`
**Particle Swarm Optimization**. This is a gradient-free optimizer that maintains a swarm of model variations ("particles"). It requires an evaluation function to score each particle.

::: tip Usage
Because PSO evaluates multiple variations of the model, it does not use the standard `backward()` pass. Instead, you provide a function that calculates loss for the current state.
:::

::: code-group
```lua [Constructor]
(params: {Tensor}, config: SwarmPSOConfig) -> Optimizer
```

```lua [Config]
type SwarmPSOConfig = {
    swarmSize: number,       -- Number of particles (models)
    evalFn: (p: number) -> number, -- Callback to evaluate particle 'p' and return loss
    
    inertia: number?,        -- Velocity retention (default 0.7)
    cognitive: number?,      -- Attraction to personal best (default 1.5)
    social: number?,         -- Attraction to global best (default 1.5)
    lr: number?,             -- Velocity scaling factor
    mutationRate: number?,   -- Prob of random mutation
    mutationStrength: number?
}
```
:::

## `Metaheuristic`
A hybrid optimizer that combines gradient descent with swarm-like behavior. Unlike SwarmPSO, this **does** use gradients (`param._grad`) to guide the particles, but also maintains personal/global bests to escape local minima.

::: code-group
```lua [Constructor]
(params: {Tensor}, config: MetaheuristicConfig) -> Optimizer
```

```lua [Config]
type MetaheuristicConfig = {
    lr: number,              -- Learning rate
    swarmSize: number?,      -- (Internal simulation size)
    gradScale: number?,      -- Weight of the gradient signal
    inertia: number?,
    cognitive: number?,
    social: number?,
    mutationRate: number?
}
```
:::