# Getting Started

Welcome to **Gradien**. This guide will help you set up the library and train your first neural network.

## Installation

### `Method 1: Wally` <Badge type="tip" text="Recommended" />

If you use [Wally](https://wally.run/) for dependency management, add Gradien to your `wally.toml`.

::: code-group
```toml [wally.toml]
[dependencies]
Gradien = "eternitydevs/gradien@1.4.0-rc1"
```

```bash [Terminal]
wally install
```
:::

### `Method 2: RBXM`

1. Download the latest release from the [GitHub Releases](https://github.com/EternityDevs/Gradien/releases) page.
2. Drag and drop the `.rbxm` file into **ReplicatedStorage** in Roblox Studio.

---

## Your First Training Loop

Let's create a simple model that learns to approximate a linear function.

### `Step 1: Setup`
Require the library and define a basic dataset.

```lua
local Gradien = require(game.ReplicatedStorage.Gradien)

-- Create dummy data: Inputs (X) and Targets (Y)
-- Shape: {Features=2, Batch=5}
local X = Gradien.Tensor.fromArray({
    1, 2, 3, 4, 5, 
    1, 2, 3, 4, 5
}, {2, 5}) 

local Y = Gradien.Tensor.fromArray({
    2, 4, 6, 8, 10
}, {1, 5})
```

### `Step 2: Define Model`
We will use a `Sequential` container with `Linear` layers and `ReLU` activations.

```lua
local model = Gradien.NN.Sequential({
    Gradien.NN.Linear(2, 8),             -- Input -> Hidden
    function(_, x) return Gradien.NN.Activations.ReLU(x) end,
    Gradien.NN.Linear(8, 1)              -- Hidden -> Output
})
```

### `Step 3: Trainer`
Use the `Trainer` class to handle the training loop, backpropagation, and optimization automatically.

```lua
local trainer = Gradien.Trainer.new({
    model = model,
    optimizerFactory = function(params) 
        return Gradien.Optim.Adam(params, 1e-2) 
    end,
    loss = Gradien.NN.Losses.mse_backward,
    metric = function(pred, target)
        -- Custom metric logic here
        return 0
    end
})

-- Start training for 100 epochs
trainer:fit(function()
    -- Dataloader generator
    return function() return X, Y end
end, { epochs = 100 })
```