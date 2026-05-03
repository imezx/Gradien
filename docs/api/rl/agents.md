# RL Agents <Badge type="tip" text="Module" />

Gradien provides a unified API for Reinforcement Learning agents.

## Agent Types

### `DQL` & `DoubleDQN` <Badge type="warning" text="Off-Policy" />
Deep Q-Learning algorithms. `DoubleDQN` reduces overestimation bias.

::: code-group
```lua [Config]
{
    actionDim: number,
    batchSize: number,
    gamma: number,
    epsilonStart: number?,
    epsilonEnd: number?,
    epsilonDecay: number?,
    modelFactory: () -> Module,
    optimizerFactory: (params) -> Optimizer,
    replay: ReplayBuffer?,
    targetSyncInterval: number?,
    tau: number? -- Soft update factor
}
```
:::

### `PPO` <Badge type="tip" text="On-Policy" />
Proximal Policy Optimization. Stable and efficient.

::: code-group
```lua [Config]
{
    policy: Module,
    value: Module,
    gamma: number,
    lam: number,
    clip: number,
    epochs: number,
    minBatch: number?,
    maxBuffer: number?,
    optimizerFactory: (params) -> Optimizer
}
```
:::

### `A2C` <Badge type="tip" text="On-Policy" />
Advantage Actor-Critic.

::: code-group
```lua [Config]
{
    policy: Module,
    value: Module,
    gamma: number,
    minBatch: number?,
    optimizerFactory: (params) -> Optimizer
}
```
:::

---

## Common Interface

### `:act`
::: code-group
```lua [Definition]
(state: Tensor, stepIndex: number?) -> number
```
:::

### `:observe`
::: code-group
```lua [Definition]
(transition: {state: Tensor, action: number, reward: number, nextState: Tensor, done: boolean}) -> ()
```
:::

### `:trainStep` <Badge type="info" text="Parallel" />
::: code-group
```lua [Definition]
() -> { loss: number, avgReturn: number? }?
```
:::

### `:loadParameters` (DQN only)
::: code-group
```lua [Definition]
(snapshot: any, strict: boolean?) -> ()
```
:::