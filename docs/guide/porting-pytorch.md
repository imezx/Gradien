# Porting PyTorch to Gradien <Badge type="tip" text="Example" />

This show you examples bridging **PyTorch** (running on a Python server) with **Gradien** (running on Roblox). This allows you to run massive models (LLMs, heavy RL policies) on a GPU and send the results to your Roblox game in real-time.

::: code-group
```bash [dependencies]
pip install torch fastapi uvicorn
```

```python [server]
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import uvicorn

app = FastAPI()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HOST = "0.0.0.0"
PORT = 8000
DTYPE = torch.bfloat16

print(f"Device: {DEVICE}")
print(f"Waiting for requests on http://{HOST}:{PORT}...")


class SingleInput(BaseModel):
    id: str
    data: List[float]


class BatchRequest(BaseModel):
    shape: List[int]
    batch: List[SingleInput]


class ProductionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # ex: input 10 -> output 5
        self.net = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 5))

    def forward(self, x):
        return self.net(x)


model = ProductionModel().to(DEVICE).to(dtype=DTYPE)
model.eval()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc):
    print(f"Error details: {exc.errors()}")
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.get("/")
def home():
    return {"status": "online", "message": "Gradien Bridge is Running"}


@app.post("/predict_batch")
async def predict_batch(req: BatchRequest):
    if not req.batch:
        return {"results": []}
    try:
        batch_data = [item.data for item in req.batch]
        input_tensor = (
            torch.tensor(batch_data, dtype=torch.float32).to(DEVICE).to(dtype=DTYPE)
        )
        if len(req.shape) > 1:
            true_shape = [-1] + req.shape
            input_tensor = input_tensor.view(*true_shape)
        with torch.no_grad():
            output_tensor = model(input_tensor)
        results = []
        output_data = output_tensor.to(dtype=torch.float32).cpu().numpy()
        for i, item in enumerate(req.batch):
            results.append({"id": item.id, "data": output_data[i].flatten().tolist()})
        return {"shape": list(output_tensor.shape[1:]), "results": results}
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
```

```lua [script]
local RemoteModel = require(script.Parent.Model)
local Gradien = require(game.ReplicatedStorage.Gradien)
local Tensor = Gradien.Tensor

local brain = RemoteModel.new({10}) 

brain.event.Event:Connect(function(outputTensor)
	if outputTensor then
		print("Server Output:", outputTensor._storage)
	end
end)

while true do
	task.wait()
	local data = table.create(10, 0)
	for i=1,10 do data[i] = math.random() end
	local input = Tensor.fromArray(data, {1, 10})
	brain:forward(input)
end
```

```lua [model]
local BridgeService = require(script.Parent.ModuleScript)
local Gradien = require(game.ReplicatedStorage.Gradien)

local Model = {}
Model.__index = Model

function Model.new(inputShape)
	local self = setmetatable({}, Model)
	self.inputShape = inputShape
	self.event = Instance.new("BindableEvent")
	return self
end

function Model:forward(inputTensor)
	local event = self.event
	local flatData = inputTensor._storage
	if not flatData then
		warn("inputTensor._storage is missing.")
		flatData = inputTensor.data
	end
	assert(flatData, "Model: Cannot send request. Tensor data is missing.")
	BridgeService.Predict(flatData, self.inputShape, function(resultData, resultShape)
		if not resultData then
			warn("inference failed.")
			event:Fire(nil)
			return
		end
		local outputTensor = Gradien.Tensor.fromArray(resultData, resultShape)
		event:Fire(outputTensor)
	end)
end

return Model
```

```lua [bridge]
local HttpService = game:GetService("HttpService")
local RunService = game:GetService("RunService")

local BridgeService = {}
BridgeService.__index = BridgeService

local CONFIG = {
	URL = "http://localhost:8000/predict_batch",
	BATCH_WINDOW = 0.05,
	MAX_BATCH_SIZE = 64,
}

local queue = {}
local pendingCallbacks = {} -- Map<ID, Function>
local lastSendTime = 0
local isSending = false

local function generateGUID(): string
	return HttpService:GenerateGUID(false):gsub("-", "")
end

local function flushQueue()
	if #queue == 0 or isSending then return end
	lastSendTime = os.clock()
	isSending = true
	local currentBatch = queue
	queue = {}
	local cleanBatch = {}
	for i, item in ipairs(currentBatch) do
		cleanBatch[i] = {
			id = item.id,
			data = item.data
		}
	end
	local payload = {
		shape = currentBatch[1].shape, 
		batch = cleanBatch 
	}
	task.spawn(function()
		local success, response = pcall(function()
			return HttpService:PostAsync(
				CONFIG.URL,
				HttpService:JSONEncode(payload),
				Enum.HttpContentType.ApplicationJson,
				false
			)
		end)
		if success then
			local decoded = HttpService:JSONDecode(response)
			for _, result in ipairs(decoded.results) do
				local reqId = result.id
				local callback = pendingCallbacks[reqId]
				if callback then
					callback(result.data, decoded.shape)
					pendingCallbacks[reqId] = nil
				end
			end
		else
			warn(`HTTP Failed: {response}`)
			for _, item in ipairs(currentBatch) do
				local cb = pendingCallbacks[item.id]
				if cb then cb(nil, "Server Error") end
				pendingCallbacks[item.id] = nil
			end
		end
		isSending = false
		if #queue > 0 then
			task.delay(CONFIG.BATCH_WINDOW, flushQueue)
		end
	end)
end

function BridgeService.Predict(inputData, inputShape, callback)
	local id = generateGUID()
	pendingCallbacks[id] = callback
	table.insert(queue, {
		id = id,
		data = inputData,
		shape = inputShape
	})
	if #queue >= CONFIG.MAX_BATCH_SIZE then
		flushQueue()
	end
end

RunService.PostSimulation:Connect(function()
	if #queue > 0 and (os.clock() - lastSendTime > CONFIG.BATCH_WINDOW) then
		flushQueue()
	end
end)

return BridgeService
```
:::