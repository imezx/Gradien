# The Trainer Loop <Badge type="tip" text="Feature" />

Writing custom training loops can be repetitive. The `Trainer` class abstracts the boilerplate of zeroing gradients, forward passes, loss calculation, backpropagation, and optimization steps.

## Basic Usage

::: code-group
```lua [Config]
local trainer = Gradien.Trainer.new({
    model = myModel,
    optimizerFactory = function(params) return Gradien.Optim.Adam(params) end,
    loss = Gradien.NN.Losses.mse_backward,
    
    -- Optional
    metric = function(pred, target) return 0 end,
    reportEvery = 10,
    callbacks = {
        onStep = function(ctx) print(ctx.loss) end
    }
})
```

```lua [Fit]
trainer:fit(function()
    -- Return a function that returns (X, Y) batches
    return MyDataLoader()
end, {
    epochs = 50,
    stepsPerEpoch = 100
})
```
:::

## Classification Trainer

For classification tasks, `Trainer.newClassification` provides sensible defaults (Cross Entropy Loss and Accuracy metric).

```lua
local clsTrainer = Gradien.Trainer.newClassification({
    model = myModel,
    optimizerFactory = myOptFactory,
    callbacks = myCallbacks
}, {
    smoothing = 0.1 -- Label smoothing
})
```

## Callbacks

The trainer supports a rich callback system to hook into the training process.

```lua
callbacks = {
    onStep = function(ctx) 
        -- ctx: { epoch, step, loss, metric, model, optimizer }
    end,
    onEpochEnd = function(ctx)
        print("End of epoch:", ctx.epoch)
    end,
    onBest = function(ctx)
        print("New best metric:", ctx.metric)
    end
}
```