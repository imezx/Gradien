# How to Create a Neural Network <Badge type="tip" text="Tutorial" />

This guide will walk you through creating, configuring, and training a simple model to learn a mathematical pattern.

We will teach the model the function: **`y = 2x`**.

## 1. Require the Library
Start by requiring the Gradien module.

```lua
local Gradien = require(game.ReplicatedStorage.Gradien)
```

## 2. Prepare Your Data
We will generate training data for numbers 1 through 20.
* **Inputs (X)**: The numbers 1, 2, 3...
* **Targets (Y)**: The answers 2, 4, 6...

::: code-group
```lua [Data Generation]
-- Generate 20 data points
local count = 20
local inputTable = {}
local targetTable = {}

for i = 1, count do
    inputTable[i] = i
    targetTable[i] = i * 2 -- The rule we want to learn
end

-- Create Tensors {Features, BatchSize}
local X = Gradien.Tensor.fromArray(inputTable, {1, count})
local Y = Gradien.Tensor.fromArray(targetTable, {1, count})
```
:::

## 3. Build the Model
For a simple linear relationship like `y = 2x`, a single **Linear Layer** is perfect. It learns a weight (`m`) and a bias (`c`) to solve `y = mx + c`.

::: tip Why no Hidden Layers?
Complex networks with activation functions (like ReLU) are great for complex patterns but struggle to **extrapolate** simple math to numbers they haven't seen before. A single Linear layer extrapolates perfectly to infinity.
:::

```lua
local model = Gradien.NN.Sequential({
    -- Input: 1 feature (the number)
    -- Output: 1 feature (the prediction)
    Gradien.NN.Linear(1, 1) 
})
```

## 4. Setup the Trainer
The `Trainer` handles the training loop.

::: warning Script Timeout
We add a **callback** to yield (`task.wait`) every few epochs. Without this, Your Roblox will crash.
:::

```lua
local trainer = Gradien.Trainer.new({
    model = model,
    
    -- Optimizer: AdamW often converges faster than standard Adam
    optimizerFactory = function(params) 
        return Gradien.Optim.AdamW(params, 1e-2, 1e-4) -- LR: 0.01
    end,
    
    loss = Gradien.NN.Losses.mse_backward,
    reportEvery = 50, -- Print less often to reduce spam

    callbacks = {
        onEpochEnd = function() 
            task.wait() -- Yield to prevent crash
        end
    }
})
```

## 5. Run Training
Use the `:fit()` method to start the training loop.

```lua
print("Starting training...")

trainer:fit(function()
    -- Return our full dataset batch
    return function() return X, Y end
end, {
    epochs = 500,   -- 500 iterations is enough for this simple task
    stepsPerEpoch = 15 
})

print("Training complete!")
```

## 6. Test the Model
Now let's test it on a number it has **never seen**, like 100!

```lua
local testVal = 100 -- lets try with 100!
local testInput = Gradien.Tensor.fromArray({testVal}, {1, 1})

-- Forward pass
local prediction = model:forward(testInput)
local result = prediction._storage[1]

print("Input:", testVal)
print("Prediction:", string.format("%.2f", result))
print("Expected:", testVal * 2)
```