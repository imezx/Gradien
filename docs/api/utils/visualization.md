# Visualization <Badge type="tip" text="Tools" />

## `Visual3D.render`

Renders the neural network into the 3D workspace using Parts and Beams.

::: code-group
```lua [Definition]
(model: Module, origin: CFrame, options: RenderOptions?) -> RenderHandle
```
```lua [Options]
{
    layerSizeHints: {number}?,
    maxNeuronsPerLayer: number?,
    maxBeamsPerEdge: number?,
    layerSpacing: number?,
    boxDepth: number?,
    unitHeight: number?,
    updateInterval: number?,
    mode: "weights" | "grads",
    palette: {Color3}?,
    name: string?,
    getActivations: ((model) -> {{number}})?,
    showNeuronValues: boolean?,
    valueEvery: number?,
    valueDecimals: number?,
    valueColor: Color3?,
    valueSize: Vector2?,
    sparsity: number?
}
```
```lua [Example]
Gradien.Util.Visual3D.render(networkOrAgent, CFrame, {
	layerSizeHints = {#States, HIDDEN, HIDDEN, HIDDEN, #Actions}, -- depth 3
	maxNeuronsPerLayer = 48,
	maxBeamsPerEdge = 256,
	layerSpacing = 10,
	unitHeight = 0.12,
	updateInterval = 1,
	mode = "weights", -- or "grads"
	getActivations = getActivations, -- or nil?
	showNeuronValues = true,
	valueEvery = 1,
	valueDecimals = 2,
})
```
:::

## `Visual2D.render`

Renders the network onto a UI layer.

::: code-group
```lua [Definition]
(model: Module, parent: Instance, options: RenderOptions?) -> RenderHandle
```
```lua [Options]
{
    size: UDim2?,
    position: UDim2?,
    anchorPoint: Vector2?,
    mode: "weights" | "grads",
    palette: {Color3}?,
    -- ... (similar to 3D options)
}
```
```lua [Example]
Gradien.Util.Visual2D.render(networkOrAgent, SurfaceGui, {
	layerSizeHints = {#States, HIDDEN, HIDDEN, HIDDEN, #Actions}, -- depth 3
	mode = "weights", -- or "grads"
	maxNeuronsPerLayer = 48,
	maxLinesPerEdge = 256,
	showLayerBoxes = true,
	updateInterval = 1,
})
```
:::