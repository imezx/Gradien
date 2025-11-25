# Experimental Neural Networks <Badge type="warning" text="Experimental" />

Located in `Gradien.Experimental.NN`. These are experimental neural network architectures and layers.

## `KAN` (Kolmogorov-Arnold Network)

A Kolmogorov-Arnold Network layer that uses learnable univariate functions (approximated via Radial Basis Functions) instead of fixed linear transformations.

::: code-group
```lua [Definition]
(in_features: number, out_features: number, grid_size: number?, spline_order: number?) -> Module
```
```lua [Example]
local kan = Gradien.Experimental.NN.KAN(128, 64, 5, 3)
local output = kan:forward(input)
```
:::

**Parameters:**
- `in_features` (number): Number of input features
- `out_features` (number): Number of output features
- `grid_size` (number, optional): Number of grid points for RBF approximation. Default: `5`
- `spline_order` (number, optional): Spline order (currently unused, reserved for future use). Default: `3`

**Returns:**
Returns a module table with:
- `forward(self, input: Tensor)`: Forward pass through the KAN layer
- `parameters(self)`: Returns list of learnable parameters

**Architecture:**
The KAN layer combines:
- A base linear transformation with SiLU activation
- A spline-based transformation using RBF (Radial Basis Function) approximation
- The final output is the sum of both paths

