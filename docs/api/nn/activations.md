# Activations <Badge type="info" text="Parallel" />

Activation functions introduce non-linearity to the network. Located in `Gradien.NN.Activations`.

## Standard

| Function | Definition |
| :--- | :--- |
| **`ReLU(x)`** | `max(0, x)` |
| **`Sigmoid(x)`** | `1 / (1 + exp(-x))` |
| **`Tanh(x)`** | `(exp(x) - exp(-x)) / (exp(x) + exp(-x))` |

## Probability

### `Softmax`
Converts a vector of values to a probability distribution. The elements of the output vector are in range (0, 1) and sum to 1.

::: code-group
```lua [Definition]
(logits: Tensor) -> Tensor
```
```lua [Example]
local probs = Gradien.NN.Softmax.forward(logits)
```
:::

## Advanced

### `GELU`
Gaussian Error Linear Unit. Often used in Transformers.
::: code-group
```lua [Definition]
(x: Tensor) -> Tensor
```
:::

### `LeakyReLU`
ReLU with a small slope for negative values to prevent dead neurons.
::: code-group
```lua [Definition]
(x: Tensor, alpha: number?) -> Tensor -- alpha defaults to 0.01
```
:::

### `ELU`
Exponential Linear Unit.
::: code-group
```lua [Definition]
(x: Tensor, alpha: number?) -> Tensor -- alpha defaults to 1.0
```
:::

### `SwiGLU`
Swish-Gated Linear Unit. Requires two inputs.
::: code-group
```lua [Definition]
(a: Tensor, b: Tensor) -> Tensor
```
:::

### `SwiGLUSplit`
Splits the input tensor into two halves and applies SwiGLU.
::: code-group
```lua [Definition]
(x: Tensor, hidden: number?) -> Tensor
```
:::

### `SiLU` (Swish)
`x * sigmoid(x)`
::: code-group
```lua [Definition]
(x: Tensor) -> Tensor
```
:::

### `Mish`
`x * tanh(ln(1 + exp(x)))`
::: code-group
```lua [Definition]
(x: Tensor) -> Tensor
```
:::

### `SeLU`
Scaled Exponential Linear Unit.
::: code-group
```lua [Definition]
(x: Tensor, alpha: number?, lambda: number?) -> Tensor
```
:::