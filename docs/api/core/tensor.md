# Tensor <Badge type="tip" text="Core" />

The `Tensor` is the fundamental data structure in Gradien. It represents a multi-dimensional array and supports automatic differentiation.

## Constructors

### `.zeros` <Badge type="info" text="Parallel" />
Creates a new tensor filled with zeros.
::: code-group
```lua [Definition]
(shape: {number}, dtype: "f32"|"f64"|"i32"?, requiresGrad: boolean?) -> Tensor
```
:::

### `.ones` <Badge type="info" text="Parallel" />
Creates a new tensor filled with ones.
::: code-group
```lua [Definition]
(shape: {number}, dtype: "f32"|"f64"|"i32"?, requiresGrad: boolean?) -> Tensor
```
:::

### `.fromArray`
Creates a tensor from a flat table.
::: code-group
```lua [Definition]
(data: {number}, shape: {number}, dtype: "f32"|"f64"?, requiresGrad: boolean?) -> Tensor
```
:::

### `.randn` <Badge type="info" text="Parallel" />
Creates a tensor with random numbers from a normal distribution (mean 0, std 1).
::: code-group
```lua [Definition]
(shape: {number}, dtype: "f32"|"f64"?, requiresGrad: boolean?) -> Tensor
```
:::

### `.empty`
Creates an uninitialized tensor (allocated but not zeroed).
::: code-group
```lua [Definition]
(shape: {number}, dtype: "f32"|"f64"?, requiresGrad: boolean?) -> Tensor
```
:::

---

## Methods

### `:reshape`
Returns a new tensor with the same data but a different shape.
::: code-group
```lua [Definition]
(self: Tensor, newShape: {number}) -> Tensor
```
:::

### `:expand` <Badge type="info" text="Parallel" />
Returns a new view of the tensor with singleton dimensions expanded to a larger size.
::: code-group
```lua [Definition]
(self: Tensor, newShape: {number}) -> Tensor
```
:::

### `:transpose` <Badge type="info" text="Parallel" />
Permutes two dimensions of the tensor.
::: code-group
```lua [Definition]
(self: Tensor, dim1: number?, dim2: number?) -> Tensor
```
:::

### `:slice` <Badge type="info" text="Parallel" />
Extracts a sub-tensor from the given dimension.
::: code-group
```lua [Definition]
(self: Tensor, dim: number, startIdx: number, endIdx: number?, step: number?) -> Tensor
```
:::

### `:narrow`
Returns a new tensor that is a narrowed version of the input tensor along dimension `dim`.
::: code-group
```lua [Definition]
(self: Tensor, dim: number, startIdx: number, length: number) -> Tensor
```
:::

### `:sum` <Badge type="info" text="Parallel" />
Returns the sum of all elements in the input tensor.
::: code-group
```lua [Definition]
(self: Tensor, dim: number?) -> Tensor
```
:::

### `:contiguous`
Returns a contiguous in memory tensor containing the same data as self tensor.
::: code-group
```lua [Definition]
(self: Tensor) -> Tensor
```
:::

### `:is_contiguous`
Returns True if the tensor is contiguous in memory in C order.
::: code-group
```lua [Definition]
(self: Tensor) -> boolean
```
:::

### `:detach`
Returns a new Tensor, detached from the current graph. The result will never require gradient.
::: code-group
```lua [Definition]
(self: Tensor) -> Tensor
```
:::

### `:noGrad`
Disables gradient recording for this specific tensor instance.
::: code-group
```lua [Definition]
(self: Tensor) -> ()
```
:::