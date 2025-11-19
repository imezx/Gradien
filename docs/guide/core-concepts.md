# Core Concepts

## Tensors <Badge type="tip" text="Core" />

The `Tensor` is the fundamental data structure in Gradien. It is a multi-dimensional array that supports automatic differentiation and parallelized operations.

## Autograd (The Tape)

Gradien uses a **Tape-based** autograd system. Operations on tensors that have `requiresGrad = true` are recorded in a graph. When you call `Tape.backwardFrom`, gradients flow backwards through this graph.

```lua
local x = G.Tensor.fromArray({2, 3}, {2, 1}, "f32", true)
local W = G.Tensor.ones({2, 2}, "f32", true)

-- Forward pass
local y = G.Math.matmul(W, x)

-- Backward pass
local grad = G.Tensor.ones(y._shape)
G.Autograd.Tape.backwardFrom(y, grad)

-- Access gradients
print(W._grad)
```