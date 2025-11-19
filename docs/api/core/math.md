# Math Operations <Badge type="info" text="Parallel" />

Located in `Gradien.Ops.Math` and `Gradien.Ops.BLAS`. These operations are parallelized.

## Element-wise

All element-wise operations support broadcasting semantics implicitly if implemented in the kernel.

### `.add`
::: code-group
```lua [Definition]
(A: Tensor, B: Tensor) -> Tensor
```
:::

### `.sub`
::: code-group
```lua [Definition]
(A: Tensor, B: Tensor) -> Tensor
```
:::

### `.mul`
::: code-group
```lua [Definition]
(A: Tensor, B: Tensor) -> Tensor
```
:::

### `.scalarAdd`
Adds a scalar value `s` to every element of `A`.
::: code-group
```lua [Definition]
(A: Tensor, s: number) -> Tensor
```
:::

### `.scalarMul`
Multiplies every element of `A` by scalar `s`.
::: code-group
```lua [Definition]
(A: Tensor, s: number) -> Tensor
```
:::

## Linear Algebra

### `.matmul`
Performs Matrix Multiplication.
::: code-group
```lua [Definition]
(A: Tensor, B: Tensor) -> Tensor
```
:::