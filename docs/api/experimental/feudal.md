# Feudal Networks

Implementation of Feudal Reinforcement Learning agent hierarchy.

## Feudal Agent

Creates a hierarchical agent with Manager and Worker levels.

```lua
local Feudal = Gradien.Experimental.RL.Feudal

local agent = Feudal({
    inputDim = 32,
    actionDim = 4,
    hiddenDim = 64,
    goalDim = 64, -- Optional, defaults to hiddenDim
    horizon = 10, -- Manager update horizon
})
```

### Configuration

- `inputDim` (number): Dimension of input state
- `actionDim` (number): Number of discrete actions
- `hiddenDim` (number): Hidden dimension for LSTM/Linear layers
- `goalDim` (number, optional): Dimension of goal vectors
- `horizon` (number, optional): Steps between Manager goal updates
- `gammaW` (number, optional): Worker discount factor
- `gammaM` (number, optional): Manager discount factor
- `lr` (number, optional): Learning rate
- `optimizerFactory` (function): Factory function to create optimizer

### Methods

- `reset(batchSize)`: Reset agent state
- `act(state)`: Select action and current goal
- `observe(transition)`: Store transition in buffers
- `trainStep()`: Perform training update (placeholder)

