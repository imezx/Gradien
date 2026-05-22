# Mamba <Badge type="warning" text="Experimental" />

Implementation of the Mamba state space model architecture.

## MambaBlock

Creates a Mamba block with selective scan mechanism.

```lua
local MambaBlock = Gradien.Experimental.NN.MambaBlock

-- Args: dModel, dState (default 16), dConv (default 4), expand (default 2)
local block = MambaBlock(64, 16, 4, 2)
local output = block:forward(input)
```

### Parameters

- `dModel` (number): Model dimension
- `dState` (number, optional): State dimension. Default: 16
- `dConv` (number, optional): Conv kernel size. Default: 4
- `expand` (number, optional): Expansion factor. Default: 2

### Returns

Returns a module table with:
- `forward(self, input)`: Forward pass
- `parameters(self)`: Returns list of parameters

