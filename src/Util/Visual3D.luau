--!optimize 2
local Visual3D = {}

local Workspace = game:GetService("Workspace")
local RunService = game:GetService("RunService")
local HttpService = game:GetService("HttpService")

export type Tensor = { _storage: {number}, _shape: {number}, _grad: {_storage:{number}}? }
export type RenderOptions = {
	layerSizeHints: {number}?,
	maxNeuronsPerLayer: number?,
	maxBeamsPerEdge: number?,
	layerSpacing: number?,
	boxDepth: number?,
	unitHeight: number?,
	updateInterval: number?,
	mode: string?,
	palette: {Color3}?,
	name: string?,
	getActivations: ((model:any)->{{number}})?,
	showNeuronValues: boolean?,
	valueEvery: number?,
	valueDecimals: number?,
	valueColor: Color3?,
	valueSize: Vector2?,
}

export type RenderHandle = {
	folder: Folder,
	setMode: (self: RenderHandle, mode: string)->(),
	update: (self: RenderHandle)->(),
	destroy: (self: RenderHandle)->(),
	setAutoUpdate: (self: RenderHandle, on: boolean)->(),
}

local function fmtVal(a: number, digits: number): string
	local aa = math.abs(a)
	if aa < 1e-3 and aa ~= 0 then
		return `{string.format("%.2e", a)}`
	end
	return string.format("%."..digits.."f", a)
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
	tl.Size = UDim2.fromScale(1,1); tl.BackgroundTransparency = 1
	tl.Text = text; tl.TextScaled = true; tl.Font = Enum.Font.GothamMedium
	tl.TextColor3 = Color3.new(1,1,1)
	tl.TextStrokeTransparency = 0.6
	tl.Parent = bb
	bb.Parent = part
end

local function labelVal(node: BasePart, valueDigits: string, valueColor: Color3, valueSize: Vector2)
	local bb = Instance.new("BillboardGui")
	bb.Name = "Val"
	bb.AlwaysOnTop = true
	bb.LightInfluence = 0
	bb.Size = UDim2.fromScale(valueSize.X, valueSize.Y)
	bb.StudsOffset = Vector3.new(0, 0, 0.28)
	local tl = Instance.new("TextLabel")
	tl.BackgroundTransparency = 1
	tl.Size = UDim2.fromScale(1,1)
	tl.TextScaled = true
	tl.Font = Enum.Font.GothamMedium
	tl.TextColor3 = valueColor
	tl.TextStrokeTransparency = 0.35
	tl.Text = fmtVal(0, valueDigits)
	tl.Parent = bb
	bb.Parent = node
	return tl
end

local function attach(part: BasePart, localPos: Vector3): Attachment
	local a = Instance.new("Attachment")
	a.CFrame = CFrame.new(localPos)
	a.Parent = part
	return a
end

local function clamp(v:number, lo:number, hi:number): number
	if v < lo then return lo elseif v > hi then return hi else return v end
end

local function ema(prev:number, x:number, alpha:number): number
	return alpha * x + (1 - alpha) * prev
end

local function maxAbs(buf:{number}): number
	local m = 0
	for i=1,#buf do
		local a = math.abs(buf[i]); if a > m then m = a end
	end
	return (m > 1e-12) and m or 1e-12
end

local function pickEven(N:number, k:number): {number}
	k = math.min(k, N)
	local out = table.create(k)
	for i=1,k do
		local idx = math.floor((i-0.5) * (N/k)) + 1
		if idx > N then idx = N end
		out[i] = idx
	end
	return out
end

local function collectWeightMats(params:{Tensor}): {Tensor}
	local Ws = {}
	for _,t in ipairs(params) do
		if t._shape and #t._shape == 2 then table.insert(Ws, t) end
	end
	return Ws
end

local function inferSizes(params:{Tensor}): {number}
	local Ws = collectWeightMats(params)
	if not next(Ws) then return {} end
	local sizes = table.create(#Ws+1)
	sizes[1] = Ws[1]._shape[2]
	for i=1,#Ws do table.insert(sizes, Ws[i]._shape[1]) end
	return sizes
end

local function sizesFromActs(getActsFn, model)
	local ok, acts = pcall(getActsFn, model)
	if ok and type(acts) == "table" and #acts >= 2 then
		local s = table.create(#acts)
		for i=1,#acts do s[i] = #acts[i] end
		return s
	end
end

function Visual3D.render(model:any, origin:CFrame, opt:RenderOptions?): RenderHandle
	opt = opt or {}
	local maxNeurons = opt.maxNeuronsPerLayer or 48
	local maxBeamsPerEdge = opt.maxBeamsPerEdge or 256
	local spacing = opt.layerSpacing or 10
	local boxDepth = opt.boxDepth or 2
	local unitH = opt.unitHeight or 0.12
	local updateEvery = opt.updateInterval or 0.8
	local sparsity = opt.sparsity or 0
	local mode = (opt.mode == "grads") and "grads" or "weights"
	local palette = opt.palette or {Color3.fromRGB(80,160,255), Color3.fromRGB(120,120,120), Color3.fromRGB(255,160,80)}
	local name = opt.name or (`NNViz_{HttpService:GenerateGUID(false)}`)
	local getActs = opt.getActivations
	local showVals = (typeof(getActs) == "function" and opt.showNeuronValues == true)
	local valueEvery = opt.valueEvery or 1
	local valueDigits = opt.valueDecimals or 2
	local valueColor = opt.valueColor or Color3.fromRGB(235,235,245)
	local valueSize = opt.valueSize or Vector2.new(0.5, 0.15)

	local folder = Instance.new("Folder")
	folder.Name = name
	folder.Parent = Workspace

	local params = model:parameters() :: {Tensor}
	local sizes = opt.layerSizeHints or (getActs and sizesFromActs(getActs, model)) or inferSizes(params)
	if #sizes < 2 then
		local p = newPart(folder, origin, Vector3.new(5,1,2), Color3.fromRGB(255,90,90), "Error")
		label(p, "Visual3D: no layers found"); 
		return {
			folder=folder,
			setMode=function() end,
			update=function() end,
			destroy=function(self) self.folder:Destroy() end,
			setAutoUpdate=function() end,
		}
	end
	local L = #sizes

	local layerParts: {BasePart} = table.create(L)
	local neuronFolders: {Folder} = table.create(L)
	local neuronPicks: { {number} } = table.create(L)
	local nodePinsByLayer: { [number]: { [number]: Attachment } } = {}
	local nodeValueLabelByLayer: { [number]: { [number]: TextLabel } } = {}

	for li=1,L do
		local n = sizes[li]
		local height = math.max(1, n * unitH)
		local color = (li==1) and palette[1] or ((li==L) and palette[3] or palette[2])
		local p = newPart(folder, origin * CFrame.new((li-1)*spacing, height/2, 0), Vector3.new(3, height, boxDepth), color, ("Layer_%d"):format(li))
		local tag = (li==1 and ("Input "..n)) or (li==L and ("Output "..n)) or ("Hidden "..n)
		label(p, tag)
		layerParts[li] = p

		-- nodes
		local nf = Instance.new("Folder"); nf.Name = "Nodes_"..li; nf.Parent = folder
		neuronFolders[li] = nf
		local picks = pickEven(n, math.min(maxNeurons, n))
		neuronPicks[li] = picks
		
		local pins = nodePinsByLayer[li] or {}
		nodePinsByLayer[li] = pins
		nodeValueLabelByLayer[li] = nodeValueLabelByLayer[li] or {}

		for k = 1, #picks do
			local j = picks[k]
			local y = -p.Size.Y/2 + (j-0.5)*(p.Size.Y / n)
			local node = Instance.new("Part")
			node.Name = ("n_%d"):format(j)
			node.Shape = Enum.PartType.Ball
			node.Anchored = true; node.CanCollide=false; node.CastShadow = false
			node.Material = Enum.Material.SmoothPlastic
			node.Color = Color3.fromRGB(200,200,200)
			node.Size = Vector3.new(0.35,0.35,0.35)
			node.CFrame = p.CFrame * CFrame.new(0, y, 0.6 + 0.5)
			node.Parent = nf

			if showVals and ((k - 1) % valueEvery == 0) then
				nodeValueLabelByLayer[li][j] = labelVal(node, valueDigits, valueColor, valueSize)
			end

			local pin = Instance.new("Attachment")
			pin.Name = ("pin_%d"):format(j)
			pin.Parent = node
			pins[j] = pin
		end
	end
	local Ws = collectWeightMats(params)
	local beamSpecs = {} :: {
		{ W: Tensor, pairs: { { iOut:number, jIn:number, beam: Beam } }, emaMax:number }
	}

	local linkCount = math.min(#Ws, L - 1)
	for li = 1, linkCount do
		local W = Ws[li]
		local fullInPick = neuronPicks[li] or {}
		local fullOutPick = neuronPicks[li + 1] or {}

		local targetSide = math.floor(math.sqrt(maxBeamsPerEdge))
		local strideIn = math.max(1, math.floor(#fullInPick / math.max(1, targetSide)))
		local strideOut = math.max(1, math.floor(#fullOutPick / math.max(1, targetSide)))

		local selIn, selOut = {}, {}
		for k=1,#fullInPick, strideIn do table.insert(selIn, fullInPick[k]) end
		for k=1,#fullOutPick, strideOut do table.insert(selOut, fullOutPick[k]) end

		local bf = Instance.new("Folder"); bf.Name = ("Beams_%d"):format(li); bf.Parent = folder
		local _pairs = {}
		local total = 0

		for oi=1,#selOut do
			for ij=1,#selIn do
				if total >= maxBeamsPerEdge then break end
				local iOut, jIn = selOut[oi], selIn[ij]
				local pinIn = nodePinsByLayer[li] and nodePinsByLayer[li][jIn]
				local pinOut = nodePinsByLayer[li+1] and nodePinsByLayer[li+1][iOut]
				if pinIn and pinOut then
					local b = Instance.new("Beam")
					b.FaceCamera = true
					b.Transparency = NumberSequence.new(0.15)
					b.Attachment0 = pinIn
					b.Attachment1 = pinOut
					b.Width0 = 0.01; b.Width1 = 0.01
					b.Color = ColorSequence.new(Color3.fromRGB(180,180,180))
					b.Parent = bf
					table.insert(_pairs, { iOut = iOut, jIn = jIn, beam = b })
					total += 1
				end
				if total >= maxBeamsPerEdge then break end
			end
		end

		table.insert(beamSpecs, { W = W, pairs = _pairs, emaMax = 1.0 })
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
			if m == "grads" or m == "weights" then currentMode = m end
		end,

		update = function(self)
			for _,spec in ipairs(beamSpecs) do
				task.desynchronize()
				local W = spec.W
				local inDim = W._shape[2]
				local useGrad = (currentMode == "grads")

				local curMax = 1e-12
				for _,pr in ipairs(spec.pairs) do
					local idx = (pr.iOut - 1) * inDim + pr.jIn
					local v = getValueField(W, idx, useGrad)
					local a = math.abs(v); if a > curMax then curMax = a end
				end
				spec.emaMax = ema(spec.emaMax, curMax, 0.2)
				local denom = spec.emaMax + 1e-12

				for _,pr in ipairs(spec.pairs) do
					local idx = (pr.iOut - 1) * inDim + pr.jIn
					local val = getValueField(W, idx, useGrad)
					local mag = clamp(math.abs(val) / denom, 0, 1)
					local w = 0.01 + 0.015 * mag
					local keep = (mag >= sparsity)
					task.synchronize()
					pr.beam.Enabled = keep
					if not keep then
						continue
					end
					local c = (useGrad and Color3.fromRGB(255,220,90))
						or ((val >= 0) and Color3.fromRGB(90,180,255) or Color3.fromRGB(255,120,120))
					pr.beam.Width0 = w; pr.beam.Width1 = w
					pr.beam.Color = ColorSequence.new(c)
				end
			end

			if getActs then
				task.desynchronize()
				local acts = getActs(model)
				if type(acts) == "table" then
					for li=1,#sizes do
						local layerActs = acts[li]
						if layerActs then
							local nf = neuronFolders[li]
							local picks = neuronPicks[li]
							for k=1,#picks do
								local j = picks[k]
								local node = nf:FindFirstChild(("n_%d"):format(j)) :: BasePart
								if node then
									local a = layerActs[j] or 0
									local s = clamp(math.abs(a), 0, 1)
									task.synchronize()
									node.Color = (a >= 0) and Color3.fromRGB(150, 220, 255) or Color3.fromRGB(255, 160, 160)
									node.Transparency = 1 - (0.15 + 0.75 * s)
									node.Size = Vector3.new(0.30 + 0.25*s, 0.30 + 0.25*s, 0.30 + 0.25*s)
									if showVals then
										local tbl = nodeValueLabelByLayer[li]
										local lbl = tbl and tbl[j]
										if lbl then
											lbl.Text = fmtVal(a, valueDigits)
										end
									end
								end
							end
						end
					end
				end
			end
		end,

		setAutoUpdate = function(self, on:boolean)
			if autoConn then autoConn:Disconnect(); autoConn = nil end
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
		
		setSparsity = function(_, th: number)
			sparsity = math.clamp(th, 0, 0.5)
		end,
		setValuesVisible = function(_, on: boolean)
			if not getActs then
				return
			end
			showVals = on
			for _, layerMap in nodeValueLabelByLayer do
				for _, tl in layerMap do
					local bb = tl and tl.Parent
					if bb and bb:IsA("BillboardGui") then
						bb.Enabled = on
					end
				end
			end
		end,

		destroy = function(self)
			if autoConn then autoConn:Disconnect(); autoConn = nil end
			self.folder:Destroy()
		end,
	}

	handle:update()
	task.synchronize()
	handle:setAutoUpdate(true)
	return handle
end

return Visual3D
