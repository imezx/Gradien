# Initializers <Badge type="info" text="Parallel" />

Located in `Gradien.Init`. Functions to initialize tensor weights.

| Function | Description |
| :--- | :--- |
| `xavierUniform(W)` | Uniform initialization for Sigmoid/Tanh layers. |
| `kaimingUniform(W)` | Uniform initialization for ReLU layers (He Init). |
| `heUniform(W)` | Alias for `kaimingUniform`. |
| `heNormal(W)` | Normal distribution for ReLU layers. |
| `lecunUniform(W)` | Efficient backprop initialization. |
| `lecunNormal(W)` | Normal distribution variant of LeCun. |
| `zeros(W)` | Fills tensor with zeros. |