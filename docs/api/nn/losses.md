# Loss Functions <Badge type="info" text="Parallel" />

Located in `Gradien.NN.Losses`. Loss functions measure how far the model's predictions are from the targets.

## Regression

### `mse_backward`
Mean Squared Error.
::: code-group
```lua [Definition]
(pred: Tensor, target: Tensor) -> (number, Tensor)
```
:::

### `l1_backward`
Mean Absolute Error (L1).
::: code-group
```lua [Definition]
(pred: Tensor, target: Tensor) -> (number, Tensor)
```
:::

### `huber_backward`
Less sensitive to outliers than MSE.
::: code-group
```lua [Definition]
(pred: Tensor, target: Tensor, delta: number?) -> (number, Tensor)
```
:::

## Classification

### `cross_entropy_backward`
Combines Softmax and Cross Entropy. Expects raw logits.
::: code-group
```lua [Definition]
(logits: Tensor, targetIndices: {number}, smoothing: number?) -> (number, Tensor)
```
:::

### `bceWithLogits_backward`
Binary Cross Entropy with Sigmoid built-in.
::: code-group
```lua [Definition]
(logits: Tensor, targets: {number}) -> (number, Tensor)
```
:::

### `focal_backward`
Focuses training on hard examples. Useful for class imbalance.
::: code-group
```lua [Definition]
(logits: Tensor, targetIdx: {number}, alpha: number?, gamma: number?) -> (number, Tensor)
```
:::

### `kl_div_backward`
Kullback-Leibler divergence.
::: code-group
```lua [Definition]
(pred: Tensor, target: Tensor, fromLogits: boolean?) -> (number, Tensor)
```
:::