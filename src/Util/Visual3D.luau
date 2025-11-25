--!native
--!optimize 2
local Visual3D = {}

local Workspace = game:GetService("Workspace")
local RunService = game:GetService("RunService")
local HttpService = game:GetService("HttpService")

local Types = require("../Types")

type Tensor = Types.Tensor
type Module = Types.Module

export type RenderOptions = {
	layerSizeHints: { number }?,
	maxNeuronsPerLayer: number?,
	maxBeamsPerEdge: number?,
	layerSpacing: number?,
	boxDepth: number?,
	unitHeight: number?,
	updateInterval: number?,
	mode: string?,
	palette: { Color3 }?,
	name: string?,
	getActivations: ((model: any) -> { { number } })?,
	showNeuronValues: boolean?,
	valueEvery: number?,
	valueDecimals: number?,
	valueColor: Color3?,
	valueSize: Vector2?,
	sparsity: number?,
}

export type RenderHandle = {
	folder: Folder,
	setMode: (self: RenderHandle, mode: string) -> (),
	update: (self: RenderHandle) -> (),
	destroy: (self: RenderHandle) -> (),
	setAutoUpdate: (self: RenderHandle, on: boolean) -> (),
	setSparsity: (self: RenderHandle, th: number) -> (),
	setValuesVisible: (self: RenderHandle, on: boolean) -> (),
}

type BeamObj = {
	beam: Beam,
	iOut: number,
	jIn: number,
	_cWidth: number,
	_cColor: Color3,
	_cEnabled: boolean,
}

type NodeObj = {
	part: BasePart,
	label: TextLabel?,
	idx: number,
	_cColor: Color3,
	_cTrans: number,
	_cSize: Vector3,
	_cText: string?,
}

local function fmtVal(a: number, digits: number): string
	local aa = math.abs(a)
	if aa < 1e-3 and aa ~= 0 then
		return `{string.format("%.2e", a)}`
	end
	return string.format("%." .. digits .. "f", a)
end

local function newPart(parent: Instance, cf: CFrame, size: Vector3, color: Color3, name: string): BasePart
	local p = Instance.new("Part")
	p.Name = name
	p.Anchored = true
	p.CanCollide = false
	p.CastShadow = false
	p.Material = Enum.Material.SmoothPlastic
	p.Color = color
	p.CFrame = cf
	p.Size = size
	p.Parent = parent
	return p
end

local function label(part: BasePart, text: string)
	local bb = Instance.new("BillboardGui")
	bb.Name = "Label"
	bb.AlwaysOnTop = true
	bb.Size = UDim2.fromScale(6, 2)
	bb.StudsOffset = Vector3.new(0, part.Size.Y * 0.6 + 0.75, 0)
	local tl = Instance.new("TextLabel")
	tl.Size = UDim2.fromScale(1, 1)
	tl.BackgroundTransparency = 1
	tl.Text = text
	tl.TextScaled = true
	tl.Font = Enum.Font.GothamMedium
	tl.TextColor3 = Color3.new(1, 1, 1)
	tl.TextStrokeTransparency = 0.6
	tl.Parent = bb
	bb.Parent = part
end

local function labelVal(node: BasePart, valueDigits: number, valueColor: Color3, valueSize: Vector2)
	local bb = Instance.new("BillboardGui")
	bb.Name = "Val"
	bb.AlwaysOnTop = true
	bb.LightInfluence = 0
	bb.Size = UDim2.fromScale(valueSize.X, valueSize.Y)
	bb.StudsOffset = Vector3.new(0, 0, 0.28)
	local tl = Instance.new("TextLabel")
	tl.BackgroundTransparency = 1
	tl.Size = UDim2.fromScale(1, 1)
	tl.TextScaled = true
	tl.Font = Enum.Font.GothamMedium
	tl.TextColor3 = valueColor
	tl.TextStrokeTransparency = 0.35
	tl.Text = fmtVal(0, valueDigits)
	tl.Parent = bb
	bb.Parent = node
	return tl
end

local function ema(prev: number, x: number, alpha: number): number
	return alpha * x + (1 - alpha) * prev
end

local function pickEven(N: number, k: number): { number }
	k = math.min(k, N)
	local out = table.create(k)
	for i = 1, k do
		local idx = math.floor((i - 0.5) * (N / k)) + 1
		if idx > N then
			idx = N
		end
		out[i] = idx
	end
	return out
end

local function collectWeightMats(params: { Tensor }): { Tensor }
	local Ws = {}
	for _, t in ipairs(params) do
		if t._shape and #t._shape == 2 then
			table.insert(Ws, t)
		end
	end
	return Ws
end

local function inferSizes(params: { Tensor }): { number }
	local Ws = collectWeightMats(params)
	local L = #Ws
	if L == 0 then
		return {}
	end
	local sizes = table.create(L + 1)
	sizes[1] = Ws[1]._shape[2]
	for i = 1, L do
		table.insert(sizes, Ws[i]._shape[1])
	end
	return sizes
end

local function sizesFromActs(getActsFn, model)
	local acts = getActsFn(model)
	local L = type(acts) == "table" and #acts or 0
	if L > 1 then
		local s = table.create(L)
		for i = 1, L do
			s[i] = #acts[i]
		end
		return s
	end
	return {}
end

function Visual3D.render(model: any, origin: CFrame, opts: RenderOptions?): RenderHandle
	local opt = opts or {}
	local maxNeurons = opt.maxNeuronsPerLayer or 48
	local maxBeamsPerEdge = opt.maxBeamsPerEdge or 256
	local spacing = opt.layerSpacing or 10
	local boxDepth = opt.boxDepth or 2
	local unitH = opt.unitHeight or 0.12
	local updateEvery = opt.updateInterval or 0.8
	local sparsity = opt.sparsity or 0
	local mode = (opt.mode == "grads") and "grads" or "weights"
	local palette = opt.palette
		or { Color3.fromRGB(80, 160, 255), Color3.fromRGB(120, 120, 120), Color3.fromRGB(255, 160, 80) }
	local name = opt.name or `NNViz_{HttpService:GenerateGUID(false)}`
	local getActs = opt.getActivations
	local showVals = (typeof(getActs) == "function" and opt.showNeuronValues == true)
	local valueEvery = opt.valueEvery or 1
	local valueDigits = opt.valueDecimals or 2
	local valueColor = opt.valueColor or Color3.fromRGB(235, 235, 245)
	local valueSize = opt.valueSize or Vector2.new(0.5, 0.15)

	local folder = Instance.new("Folder")
	folder.Name = name
	folder.Parent = Workspace

	local params = model:parameters() :: { Tensor }
	local sizes = opt.layerSizeHints or (getActs and sizesFromActs(getActs, model)) or inferSizes(params)

	local L = #sizes
	if L < 2 then
		local p = newPart(folder, origin, Vector3.new(5, 1, 2), Color3.fromRGB(255, 90, 90), "Error")
		label(p, "Visual3D: < 2 layers detected")
		return {
			folder = folder,
			setMode = function() end,
			update = function() end,
			destroy = function(self)
				self.folder:Destroy()
			end,
			setAutoUpdate = function() end,
			setSparsity = function() end,
			setValuesVisible = function() end,
		}
	end
	local layerParts: { BasePart } = table.create(L)
	local nodeCache: { { NodeObj } } = table.create(L)
	local neuronPicks: { { number } } = table.create(L)
	local nodePinsByLayer: { [number]: { [number]: Attachment } } = {}
	for li = 1, L do
		local n = sizes[li]
		local height = math.max(1, n * unitH)
		local color = (li == 1) and palette[1] or ((li == L) and palette[3] or palette[2])
		local p = newPart(
			folder,
			origin * CFrame.new((li - 1) * spacing, height / 2, 0),
			Vector3.new(3, height, boxDepth),
			color,
			`Layer_{li}`
		)
		local tag = (li == 1 and `Input {n}`) or (li == L and `Output {n}`) or `Hidden {n}`
		label(p, tag)
		layerParts[li] = p

		local nf = Instance.new("Folder")
		nf.Name = `Nodes_{li}`
		nf.Parent = folder
		local picks = pickEven(n, math.min(maxNeurons, n))
		neuronPicks[li] = picks
		local layerNodes = table.create(#picks)
		nodeCache[li] = layerNodes
		local pins = {}
		nodePinsByLayer[li] = pins

		for k = 1, #picks do
			local j = picks[k]
			local y = -p.Size.Y / 2 + (j - 0.5) * (p.Size.Y / n)
			local node = Instance.new("Part")
			node.Name = `n_{j}`
			node.Shape = Enum.PartType.Ball
			node.Anchored = true
			node.CanCollide = false
			node.CastShadow = false
			node.Material = Enum.Material.SmoothPlastic
			node.Color = Color3.fromRGB(200, 200, 200)
			node.Size = Vector3.new(0.35, 0.35, 0.35)
			node.CFrame = p.CFrame * CFrame.new(0, y, 0.6 + 0.5)
			node.Parent = nf
			local lbl: TextLabel? = nil
			if showVals and ((k - 1) % valueEvery == 0) then
				lbl = labelVal(node, valueDigits, valueColor, valueSize)
			end
			local pin = Instance.new("Attachment")
			pin.Name = `pin_{j}`
			pin.Parent = node
			pins[j] = pin
			layerNodes[k] = {
				part = node,
				label = lbl,
				idx = j,
				_cColor = Color3.new(1, 1, 1),
				_cTrans = 0,
				_cSize = Vector3.new(1, 1, 1),
				_cText = nil,
			}
		end
	end
	local Ws = collectWeightMats(params)
	local beamSpecs = {} :: { { W: Tensor, objects: { BeamObj }, emaMax: number } }
	local linkCount = math.min(#Ws, L - 1)
	for li = 1, linkCount do
		local W = Ws[li]
		local fullInPick = neuronPicks[li] or {}
		local fullOutPick = neuronPicks[li + 1] or {}
		local targetSide = math.floor(math.sqrt(maxBeamsPerEdge))
		local strideIn = math.max(1, math.floor(#fullInPick / math.max(1, targetSide)))
		local strideOut = math.max(1, math.floor(#fullOutPick / math.max(1, targetSide)))
		local bf = Instance.new("Folder")
		bf.Name = `Beams_{li}`
		bf.Parent = folder
		local objects: { BeamObj } = {}
		local total = 0
		for oi = 1, #fullOutPick, strideOut do
			if total >= maxBeamsPerEdge then
				break
			end
			for ij = 1, #fullInPick, strideIn do
				if total >= maxBeamsPerEdge then
					break
				end
				local iOut = fullOutPick[oi]
				local jIn = fullInPick[ij]
				local pinIn = nodePinsByLayer[li] and nodePinsByLayer[li][jIn]
				local pinOut = nodePinsByLayer[li + 1] and nodePinsByLayer[li + 1][iOut]
				if pinIn and pinOut then
					local b = Instance.new("Beam")
					b.FaceCamera = true
					b.Transparency = NumberSequence.new(0.15)
					b.Attachment0 = pinIn
					b.Attachment1 = pinOut
					b.Width0 = 0.01
					b.Width1 = 0.01
					b.Color = ColorSequence.new(Color3.fromRGB(180, 180, 180))
					b.Parent = bf
					table.insert(objects, {
						beam = b,
						iOut = iOut,
						jIn = jIn,
						_cWidth = 0.01,
						_cColor = Color3.new(1, 1, 1),
						_cEnabled = true,
					})
					total += 1
				end
			end
		end
		table.insert(beamSpecs, { W = W, objects = objects, emaMax = 1.0 })
	end
	local function getValueField(t: Tensor, idx: number, useGrad: boolean): number
		if useGrad and t._grad and t._grad._storage then
			return math.abs(t._grad._storage[idx] or 0)
		end
		return t._storage[idx] or 0
	end
	local autoConn: RBXScriptConnection? = nil
	local currentMode = mode
	local handle: RenderHandle
	handle = {
		folder = folder,
		setMode = function(self, m: string)
			if m == "grads" or m == "weights" then
				currentMode = m
			end
		end,
		setSparsity = function(_, th: number)
			sparsity = math.clamp(th, 0, 0.5)
		end,
		setValuesVisible = function(_, on: boolean)
			if not getActs then
				return
			end
			showVals = on
			for _, layerList in nodeCache do
				for _, obj in layerList do
					if obj.label and obj.label.Parent and obj.label.Parent:IsA("BillboardGui") then
						obj.label.Parent.Enabled = on
					end
				end
			end
		end,
		update = function(self)
			task.desynchronize()
			local useGrad = (currentMode == "grads")
			for _, spec in ipairs(beamSpecs) do
				local W = spec.W
				local inDim = W._shape[2]
				local curMax = 1e-12
				for _, obj in ipairs(spec.objects) do
					local idx = (obj.iOut - 1) * inDim + obj.jIn
					local v = getValueField(W, idx, useGrad)
					local a = math.abs(v)
					if a > curMax then
						curMax = a
					end
				end
				spec.emaMax = ema(spec.emaMax, curMax, 0.2)
				local denom = spec.emaMax + 1e-12
				for _, obj in ipairs(spec.objects) do
					local idx = (obj.iOut - 1) * inDim + obj.jIn
					local val = getValueField(W, idx, useGrad)
					local mag = math.clamp(math.abs(val) / denom, 0, 1)
					local keep = (mag >= sparsity)
					obj._cEnabled = keep
					if keep then
						local w = 0.01 + 0.015 * mag
						obj._cWidth = w
						if useGrad then
							obj._cColor = Color3.fromRGB(255, 220, 90)
						else
							if val >= 0 then
								obj._cColor = Color3.fromRGB(90, 180, 255)
							else
								obj._cColor = Color3.fromRGB(255, 120, 120)
							end
						end
					end
				end
			end
			local actResults = nil
			if getActs then
				local acts = getActs(model)
				if type(acts) == "table" then
					actResults = acts
					for li = 1, #nodeCache do
						local layerActs = acts[li]
						local layerNodes = nodeCache[li]
						if layerActs and layerNodes then
							for _, nodeObj in ipairs(layerNodes) do
								local j = nodeObj.idx
								local a = layerActs[j] or 0
								local s = math.clamp(math.abs(a), 0, 1)
								if a >= 0 then
									nodeObj._cColor = Color3.fromRGB(150, 220, 255)
								else
									nodeObj._cColor = Color3.fromRGB(255, 160, 160)
								end
								nodeObj._cTrans = 1 - (0.15 + 0.75 * s)
								local sz = 0.30 + 0.25 * s
								nodeObj._cSize = Vector3.new(sz, sz, sz)
								if showVals and nodeObj.label then
									nodeObj._cText = fmtVal(a, valueDigits)
								end
							end
						end
					end
				end
			end
			task.synchronize()
			for _, spec in ipairs(beamSpecs) do
				for _, obj in ipairs(spec.objects) do
					obj.beam.Enabled = obj._cEnabled
					if obj._cEnabled then
						local w = obj._cWidth
						obj.beam.Width0 = w
						obj.beam.Width1 = w
						obj.beam.Color = ColorSequence.new(obj._cColor)
					end
				end
			end
			if getActs and actResults then
				for li = 1, #nodeCache do
					local layerNodes = nodeCache[li]
					for _, nodeObj in ipairs(layerNodes) do
						nodeObj.part.Color = nodeObj._cColor
						nodeObj.part.Transparency = nodeObj._cTrans
						nodeObj.part.Size = nodeObj._cSize
						if nodeObj._cText and nodeObj.label then
							nodeObj.label.Text = nodeObj._cText
						end
					end
				end
			end
		end,
		setAutoUpdate = function(self, on: boolean)
			if autoConn then
				autoConn:Disconnect()
				autoConn = nil
			end
			if on then
				local t = 0.0
				autoConn = RunService.PreSimulation:Connect(function(dt: number)
					t += dt
					if t >= updateEvery then
						t = 0.0
						self:update()
					end
				end)
			end
		end,
		destroy = function(self)
			if autoConn then
				autoConn:Disconnect()
				autoConn = nil
			end
			self.folder:Destroy()
		end,
	}
	handle:update()
	task.synchronize()
	handle:setAutoUpdate(true)
	return handle
end

return Visual3D
