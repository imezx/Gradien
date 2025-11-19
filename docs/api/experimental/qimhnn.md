# QIMHNN <Badge type="warning" text="Experimental" />

**Quantum-Inspired Metaheuristic Neural Network**

This module implements a neural network architecture inspired by quantum mechanics concepts (Superposition, Interference, and Entanglement) to potentially escape local minima and improve generalization.

## Constructor

Located in `Gradien.Experimental.Models.QIMHNN`.

::: code-group
```lua [Definition]
(config: QIMHNNConfig) -> Module
```

```lua [Configuration]
type QIMHNNConfig = {
    inputDim: number,
    outputDim: number,
    hiddenDims: {number}?,     -- e.g. {64, 64}
    
    -- Quantum Features
    useSuperposition: boolean?, -- Default: true
    useInterference: boolean?,  -- Default: true
    useEntanglement: boolean?,  -- Default: true
    quantumAmplitude: number?,  -- Amplitude factor for |z| (default 0.1)
    
    -- Standard Features
    activation: string?,       -- Default activation (e.g. "ReLU")
    finalActivation: string?,  -- Output activation
    dropout: number?,          -- Dropout probability
    layerNorm: boolean?        -- Apply LayerNorm
}
```

```lua [Example]
local model = Gradien.Experimental.Models.QIMHNN({
    inputDim = 10,
    outputDim = 2,
    hiddenDims = {32}, -- depth: 1
    useSuperposition = true,
    useInterference = true
})
```
:::

## Concepts

* **Superposition**: Weights utilize both real and imaginary components. The output incorporates a magnitude term `|z|`.
* **Interference**: Scales the output based on the phase difference between real and imaginary parts, simulating wave interference.
* **Entanglement**: Adds a dense connection `E` that mixes state information across the layer, simulating particle entanglement.