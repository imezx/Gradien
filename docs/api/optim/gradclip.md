# Gradient Clipping <Badge type="info" text="Parallel" />

Located in `Gradien.GradClip`. Used to prevent exploding gradients, especially in RNNs or deep networks.

## Functions

### `.clipValue`
Clips gradient values element-wise to be within `[-clip, clip]`.
::: code-group
```lua [Definition]
(params: {Tensor}, clip: number) -> ()
```
:::

### `.clipNorm`
Scales gradients so that their total norm does not exceed `maxNorm`. This preserves the direction of the gradient.
::: code-group
```lua [Definition]
(params: {Tensor}, maxNorm: number) -> ()
```
:::