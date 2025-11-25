# Layers <Badge type="tip" text="Module" />

Layers are the building blocks of neural networks. They store learnable parameters (`weights` and `biases`).

## `NN.Linear`

Applies a linear transformation to the incoming data: `y = x * W^T + b`.

::: code-group
```lua [Definition]
(
    inFeatures: number,
    outFeatures: number,
    initializer: ((fanIn, fanOut) -> number)?
) -> Module
```
```lua [Example]
local layer = Gradien.NN.Linear(128, 64)
```
:::

## `NN.Sequential`

A container that chains modules together. Data flows through them in the order they are defined.

::: code-group
```lua [Definition]
(layers: { Module | (Tensor)->Tensor }) -> Module
```
```lua [Example]
local model = Gradien.NN.Sequential({
    Gradien.NN.Linear(10, 20),
    Gradien.NN.Linear(20, 5)
})
```
:::

## `NN.Conv2d`

Applies a 2D convolution over an input signal composed of several input planes.

::: code-group
```lua [Definition]
(C_in: number, C_out: number, KH: number, KW: number) -> Module
```
```lua [Example]
local conv = Gradien.NN.Conv2d(3, 64, 3, 3) -- 3 input channels, 64 output, 3x3 kernel
```
:::

## `NN.MaxPool2d`

Applies a 2D max pooling over an input signal.

::: code-group
```lua [Definition]
(KH: number, KW: number, stride: number) -> Module
```
```lua [Example]
local pool = Gradien.NN.MaxPool2d(2, 2, 2) -- 2x2 kernel, stride 2
```
:::

## `NN.AvgPool2d`

Applies a 2D average pooling over an input signal.

::: code-group
```lua [Definition]
(KH: number, KW: number, stride: number) -> Module
```
```lua [Example]
local pool = Gradien.NN.AvgPool2d(2, 2, 2) -- 2x2 kernel, stride 2
```
:::

## `NN.ConvTranspose2d`

Applies a 2D transposed convolution operator over an input image composed of several input planes.

::: code-group
```lua [Definition]
(C_in: number, C_out: number, KH: number, KW: number) -> Module
```
```lua [Example]
local convT = Gradien.NN.ConvTranspose2d(64, 3, 3, 3) -- 64 input channels, 3 output, 3x3 kernel
```
:::

## `NN.GroupNorm`

Applies Group Normalization over a mini-batch of inputs. Divides channels into groups and normalizes within each group independently.

::: code-group
```lua [Definition]
(num_groups: number, num_channels: number, eps: number?) -> Module
```
```lua [Example]
local gn = Gradien.NN.GroupNorm(8, 64) -- 8 groups, 64 channels
```
:::

**Parameters:**
- `num_groups` (number): Number of groups to divide channels into. Must divide `num_channels` evenly.
- `num_channels` (number): Number of channels expected in the input.
- `eps` (number, optional): Small value added to variance for numerical stability. Default: `1e-5`.