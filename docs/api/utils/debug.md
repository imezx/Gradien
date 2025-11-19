# Debugging & Metrics <Badge type="tip" text="Tools" />

## Metrics <Badge type="info" text="Parallel" />

Located in `Gradien.Metrics`.

* **`accuracy(logits, targets)`**: Returns classification accuracy (0.0 - 1.0).
* **`topk(logits, targets, k)`**: Returns top-k accuracy.
* **`mse(pred, target)`**: Mean Squared Error.
* **`prf1(pred, target, C)`**: Returns Precision, Recall, and F1 Score.
* **`confusion(pred, target, C)`**: Returns a Confusion Matrix `{ {number} }`.

## Debug Tools

Located in `Gradien.Debug` and `Gradien.Util.Anomaly`.

### `Anomaly`
Low-level checks for numerical stability.

* **`hasNaN(t)`**: Returns true if tensor contains `NaN`.
* **`hasInf(t)`**: Returns true if tensor contains `Inf`.
* **`hasBadValues(t)`**: Returns true if `NaN` or `Inf`.

### `GradStats`
Analyzes gradients across an entire model. Useful for diagnosing vanishing/exploding gradients.

::: code-group
```lua [Definition]
GradStats.forModel(model: Module) -> { [Tensor]: StatTable }
```
```lua [StatTable]
{
    min: number,
    max: number,
    mean: number,
    std: number,
    n: number
}
```
:::

### `Debug.wrapOptimizer`
Creates a wrapper around an optimizer that automatically performs gradient clipping and NaN checks before every step.

::: code-group
```lua [Definition]
(
    opt: Optimizer, 
    params: {Tensor}, 
    cfg: { 
        maxGradNorm: number?, 
        clipValue: number?, 
        warnOnNaN: boolean?,
        label: string? 
    }
) -> WrappedOptimizer
```
:::