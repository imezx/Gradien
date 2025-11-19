# Data Pipeline <Badge type="tip" text="Utils" />

Tools for loading, scaling, and encoding data. Located in `Gradien.Preprocess`.

## Data Loading

### `DataLoader`
Creates an iterator that yields batches of data from a dataset.

::: code-group
```lua [Definition]
(
    dataset: Dataset,
    batchSize: number,
    shuffle: boolean,
    generator: { randint: (a,b)->number }?,
    drop_last: boolean?
) -> Iterator
```
:::

## Scaling & Normalization

### `StandardScaler` <Badge type="info" text="Parallel" />
Standardizes features by removing the mean and scaling to unit variance.

Formula: `z = (x - mean) / std`

::: code-group
```lua [Constructor]
() -> StandardScaler
```
```lua [Methods]
scaler:fit(X: Tensor)
local transformed = scaler:transform(X: Tensor)
```
:::

### `MinMaxScaler` <Badge type="info" text="Parallel" />
Scales features to a specific range (default 0 to 1).

Formula: `x_scaled = (x - min) / (max - min)`

::: code-group
```lua [Constructor]
(min: number?, max: number?) -> MinMaxScaler
```
```lua [Methods]
scaler:fit(X: Tensor)
local transformed = scaler:transform(X: Tensor)
```
:::

### `RunningNorm` <Badge type="info" text="Stream" />
Maintains a running mean and variance for scalar streams. Useful in Reinforcement Learning where the full dataset isn't available upfront.

::: code-group
```lua [Constructor]
(eps: number?) -> RunningNorm
```
```lua [Methods]
norm:update(x: number)      -- Updates stats with new value
norm:normalize(x: number)   -- Returns (x - mean) / std
norm:var()                  -- Current variance
norm:std()                  -- Current standard deviation
```
:::

## Encoders & Transformers

### `OneHot` <Badge type="info" text="Parallel" />
Creates a function that converts a list of class indices into a One-Hot encoded batch.

::: code-group
```lua [Definition]
(numClasses: number) -> (indices: {number}) -> Tensor
```
```lua [Example]
local encoder = Gradien.Preprocess.OneHot(10)
-- Batch of 3 samples with classes 1, 5, and 9
local batch = encoder({1, 5, 9}) 
-- Result: Tensor of shape {10, 3}
```
:::

### `PCA` (Principal Component Analysis) <Badge type="info" text="Parallel" />
Performs dimensionality reduction by projecting data onto its principal components.

::: code-group
```lua [Constructor]
(X: Tensor, K: number, iters: number?) -> PCA_Model
```
```lua [Methods]
-- Projects new data X onto the K principal components found during init
local reduced = pca:transform(X_new) 
```
:::

### `SinusoidalPE` <Badge type="info" text="Parallel" />
Adds Sinusoidal Positional Embeddings to a sequence tensor. Crucial for Transformer models to understand order.

::: code-group
```lua [Definition]
(x: Tensor, sequenceLength: number) -> Tensor
```
```lua [Example]
-- x: {EmbeddingDim, Batch * SeqLen}
local output = Gradien.Preprocess.SinusoidalPE(x, 128)
```
:::