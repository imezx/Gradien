# Model Tools <Badge type="tip" text="Utils" />

Located in `Gradien.Util`. These tools help you inspect, profile, and modify models.

## `ModelStats`

Utilities to calculate parameter counts and memory usage.

### `.summary`
Prints or returns a summary of the model size.

::: code-group
```lua [Definition]
(model: Module, printOut: boolean?) -> (number, string)
```
```lua [Example]
-- Prints: params=1250 (~10.00 KB)
local count, sizeStr = Gradien.Util.ModelStats.summary(model, true)
```
:::

### `.count`
Returns raw parameter count and byte size.

::: code-group
```lua [Definition]
(model: Module) -> (number, number)
```
:::

---

## `Profiler`

A lightweight instrumentation profiler to measure performance bottlenecks in your training loop.

### Usage

```lua
local Profiler = Gradien.Util.Profiler

-- Wrap the whole training loop in a profiler scope
local function step()
    -- ... code
end
Profiler.withEnabled(true, function()
    Profiler.scope("train_loop", function()
        for _ = 1, 500 do
            Profiler.scope("train_step", step)
        end
    end)
end)

-- Print a report
Profiler.report()
```

### Methods

* **`start(name)`**: Pushes a marker onto the stack.
* **`stop(name?)`**: Pops the marker and records time.
* **`scope(name, fn, ...)`**: Runs `fn` inside a named profile scope.
* **`report(opts)`**: Prints a formatted table of timings.

---

## `Hooks`

Allows you to inject custom logic into the forward pass of any module.

### `.addForwardHook`
Attaches a function that runs *after* the module's forward pass.

::: code-group
```lua [Definition]
(module: Module, fn: (self, output) -> ()) -> ()
```
```lua [Example]
Gradien.Util.Hooks.addForwardHook(layer, function(self, out)
    print("Layer output shape:", out._shape)
end)
```
:::

### `.removeForwardHook`
Removes the attached hook from the module.

::: code-group
```lua [Definition]
(module: Module) -> ()
```
:::