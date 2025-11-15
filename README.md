https://devforum.roblox.com/t/gradien-parallelized-machine-deep-learning/4055552

## Models

Gradien now ships with a batteries-included `Gradien.Models` namespace that covers common deep learning workloads:

- `MLP` – flexible feed-forward stacks with layer norm + dropout support.
- `ResMLP` – residual transformer-style blocks for deeper tabular mixers.
- `ConvNet` – image classifier with configurable convolutional stages and linear heads.
- `TransformerEncoder` – multi-head self-attention encoder with pre-norm and feed-forward stacks.
- `SequenceClassifier` – stacked GRU/LSTM/RNN classifier for `{features, time, batch}` tensors.
- `AutoEncoder` – symmetric encoder/decoder for representation learning.

Example:

```
local Gradien = require(path.to.gradien)

local model = Gradien.Models.MLP({
	inputDim = 128,
	outputDim = 10,
	hiddenDims = { 256, 256 },
	activation = "gelu",
	dropout = 0.1,
})
```