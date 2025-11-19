# State Management <Badge type="warning" text="Important" />

Since Roblox games rely on `DataStoreService` for persistence, Gradien provides utilities to serialize models and optimizers into simple tables that can be saved directly.

## Models

### Dumping
Converts the model's parameters into a serializable Snapshot format.

```lua
local snapshot = Gradien.State.dump(model:parameters())
-- Returns: { { shape={...}, dtype="f32", data={...} }, ... }
```

### Loading
Restores parameters from a snapshot. This modifies the model in-place.

```lua
Gradien.State.load(model:parameters(), snapshot)
```

## Binary Serialization (Buffer) <Badge type="tip" text="Recommended" />

For maximum storage efficiency, you can serialize snapshots into binary buffers. This significantly reduces the data size compared to JSON tables.

### `State.toBuffer`
Converts a snapshot table into a compact buffer.

```lua
local snapshot = Gradien.State.dump(model:parameters())
local binaryData = Gradien.State.toBuffer(snapshot)

-- Save binaryData to DataStore
-- (Ensure your DataStore implementation supports buffer or encode to base64)
```

### `State.fromBuffer`
Restores a snapshot from a binary buffer.

```lua
local snapshot = Gradien.State.fromBuffer(binaryData)
if snapshot then
    Gradien.State.load(model:parameters(), snapshot)
end
```

## Optimizers

Optimizers contain state (like momentum buffers in Adam) that must be saved to resume training effectively.

```lua
-- Save
local optState = Gradien.State.dumpOptimizer(optimizer)

-- Load
Gradien.State.loadOptimizer(optimizer, optState)
```

## Full Trainer Checkpoint

You can save the entire state of a `Trainer` instance, including the model, optimizer, current epoch, and best metric.

```lua
local checkpoint = Gradien.State.dumpTrainer(trainer)

-- Save to DataStore
DataStore:SetAsync("TrainingCheckpoint", checkpoint)

-- Load later
Gradien.State.loadTrainer(trainer, checkpoint)
```