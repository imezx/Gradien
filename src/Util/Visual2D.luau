--!optimize 2
local Visual2D = {}

local RunService = game:GetService("RunService")
local HttpService = game:GetService("HttpService")

export type Tensor = { _storage: {number}, _shape: {number}, _grad: {_storage:{number}}? }

export type RenderOptions = {
	parent: Instance?,
	size: UDim2?,
	position: UDim2?,
	anchorPoint: Vector2?,
	layerSizeHints: {number}?,
	maxNeuronsPerLayer: number?,
	maxLinesPerEdge: number?,
	padding: number?,
	colSpacing: number?,
	rowMinGap: number?,
	nodeRadius: number?,
	lineMin: number?,
	lineMax: number?,
	updateInterval: number?,
	mode: string?,
	palette: {Color3}?,
	name: string?,
	getActivations: ((model:any)->{{number}})?,
	showLayerBoxes: boolean?,
	showLabels: boolean?,
	backgroundColor: Color3?,
}

export type RenderHandle = {
	container: GuiObject,
	setMode: (self: RenderHandle, mode: string)->(),
	update: (self: RenderHandle)->(),
	setAutoUpdate: (self: RenderHandle, on: boolean)->(),
	destroy: (self: RenderHandle)->(),
}

local function clamp(v:number, lo:number, hi:number): number
	if v < lo then return lo elseif v > hi then return hi else return v end
end

local function ema(prev:number, x:number, alpha:number): number
	return alpha * x + (1 - alpha) * prev
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

local function newFrame(parent: Instance, name: string, size: UDim2, pos: UDim2, ap: Vector2, bg: Color3?, transp:number?): Frame
	local f = Instance.new("Frame")
	f.Name = name
	f.AnchorPoint = ap
	f.Size = size
	f.Position = pos
	f.BackgroundColor3 = bg or Color3.new(0,0,0)
	f.BackgroundTransparency = (transp ~= nil) and transp or 1
	f.BorderSizePixel = 0
	f.Parent = parent
	return f
end

local function newText(parent: Instance, text: string, pos: UDim2, ap: Vector2, color: Color3): TextLabel
	local t = Instance.new("TextLabel")
	t.AnchorPoint = ap
	t.BackgroundTransparency = 1
	t.Size = UDim2.fromOffset(200, 20)
	t.Position = pos
	t.Font = Enum.Font.GothamMedium
	t.TextScaled = true
	t.Text = text
	t.TextColor3 = color
	t.TextStrokeTransparency = 0.75
	t.TextXAlignment = Enum.TextXAlignment.Center
	t.Parent = parent
	return t
end

local function newCircle(parent: Instance, center: Vector2, radius:number, color: Color3): Frame
	local f = Instance.new("Frame")
	f.Name = "Node"
	f.AnchorPoint = Vector2.new(0.5, 0.5)
	f.Position = UDim2.fromOffset(center.X, center.Y)
	f.Size = UDim2.fromOffset(radius*2, radius*2)
	f.BackgroundColor3 = color
	f.BorderSizePixel = 0
	f.ZIndex = 2
	local uic = Instance.new("UICorner")
	uic.CornerRadius = UDim.new(1,0)
	uic.Parent = f
	local uis = Instance.new("UIStroke")
	uis.Parent = f
	f.Parent = parent
	return f
end

local function newLine(parent: Instance, p1: Vector2, p2: Vector2, thickness:number, color: Color3): Frame
	local dx, dy = p2.X - p1.X, p2.Y - p1.Y
	local len = math.max(1, math.sqrt(dx*dx + dy*dy))
	local mid = Vector2.new((p1.X + p2.X)/2, (p1.Y + p2.Y)/2)
	local angle = math.deg(math.atan2(dy, dx))
	local f = Instance.new("Frame")
	f.Name = "Line"
	f.AnchorPoint = Vector2.new(0.5, 0.5)
	f.Position = UDim2.fromOffset(mid.X, mid.Y)
	f.Size = UDim2.fromOffset(len, thickness)
	f.Rotation = angle
	f.BackgroundColor3 = color
	f.BorderSizePixel = 0
	f.Parent = parent
	return f
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

local function posByIndex(nodePicks, nodePos, sizes, layerIdx:number, neuronIndex:number): Vector2
	local pickList = nodePicks[layerIdx]
	local posList = nodePos[layerIdx]
	for k=1,#pickList do
		if pickList[k] == neuronIndex then return posList[k] end
	end
	local approx = math.clamp(math.floor((neuronIndex-1)/math.max(1, sizes[layerIdx]) * #posList)+1, 1, #posList)
	return posList[approx]
end

function Visual2D.render(model:any, parent: Instance, opt: RenderOptions?): RenderHandle
	opt = opt or {}
	assert(parent ~= nil, "Visual2D.render: parent (ScreenGui/BillboardGui/SurfaceGui/Frame) is required")

	local NAME = opt.name or ("NNViz2D_"..HttpService:GenerateGUID(false))
	local canvasSize = opt.size or UDim2.fromScale(1, 1)
	local canvasPos = opt.position or UDim2.fromScale(0.5, 0.5)
	local canvasAP = opt.anchorPoint or Vector2.new(0.5, 0.5)
	local padding = opt.padding or 24
	local nodeR = opt.nodeRadius or 5
	local lineMin = opt.lineMin or 1
	local lineMax = opt.lineMax or 6
	local rowMinGap = opt.rowMinGap or 6
	local maxNodes = opt.maxNeuronsPerLayer or 48
	local maxLinesPerEdge = opt.maxLinesPerEdge or 256
	local updateEvery = opt.updateInterval or 0.7
	local currentMode = (opt.mode == "grads") and "grads" or "weights"
	local palette = opt.palette or {Color3.fromRGB(80,160,255), Color3.fromRGB(120,120,120), Color3.fromRGB(255,160,80)}
	local showBoxes = (opt.showLayerBoxes ~= false)
	local showLabels = (opt.showLabels ~= false)

	local container: Frame
	if parent:IsA("ScreenGui") or parent:IsA("BillboardGui") or parent:IsA("SurfaceGui") then
		container = newFrame(parent, NAME, canvasSize, canvasPos, canvasAP, opt.backgroundColor or Color3.new(0,0,0), 1)
	elseif parent:IsA("GuiObject") then
		container = newFrame(parent, NAME, canvasSize, canvasPos, canvasAP, opt.backgroundColor or Color3.new(0,0,0), 1)
	else
		error("Visual2D.render: parent must be a ScreenGui/BillboardGui/SurfaceGui/Frame")
	end

	local params = (model.parameters and model:parameters()) or {}
	local sizes = opt.layerSizeHints or inferSizes(params)
	if #sizes < 2 then
		local msg = newText(container, "Visual2D: no layers found", UDim2.fromScale(0.5,0.5), Vector2.new(0.5,0.5), Color3.fromRGB(255,120,120))
		msg.TextSize = 16
		return {
			container = container,
			setMode = function() end,
			update = function() end,
			setAutoUpdate = function() end,
			destroy = function(self) self.container:Destroy() end,
		}
	end
	local L = #sizes

	local absW = container.AbsoluteSize.X > 0 and container.AbsoluteSize.X or (canvasSize.X.Offset > 0 and canvasSize.X.Offset or 900)
	local absH = container.AbsoluteSize.Y > 0 and container.AbsoluteSize.Y or (canvasSize.Y.Offset > 0 and canvasSize.Y.Offset or 320)
	local innerW = math.max(1, absW - padding*2)
	local innerH = math.max(1, absH - padding*2)

	local colSpacing = opt.colSpacing or (innerW / math.max(1, L-1))
	local colX = table.create(L)
	for li=1,L do colX[li] = padding + (li-1)*colSpacing end

	local layerFrames: {Frame} = {}
	local labelFrames: {TextLabel} = {}
	local nodeFolders: {Frame} = {}
	local nodePicks: { {number} } = {}
	local nodePos: { {Vector2} } = {}

	for li=1,L do
		local n = sizes[li]
		local picks = pickEven(n, math.min(maxNodes, n))
		nodePicks[li] = picks
		local count = #picks
		local totalHeight = math.max(count * (2*nodeR + rowMinGap) - rowMinGap, 1)
		local topY = padding + (innerH - totalHeight)/2

		local lf = newFrame(container, ("Layer_%d"):format(li), UDim2.fromOffset(1,1), UDim2.fromOffset(0,0), Vector2.new(0,0), nil, 1)
		layerFrames[li] = lf

		if showBoxes then
			local box = newFrame(container, ("ColBox_%d"):format(li), UDim2.fromOffset(40, innerH), UDim2.fromOffset(colX[li]-20, padding), Vector2.new(0,0), (li==1 and palette[1]) or (li==L and palette[3]) or palette[2], 0.92)
			box.BackgroundTransparency = 0.92
		end

		if showLabels then
			local tag = (li==1 and ("Input "..n)) or (li==L and ("Output "..n)) or ("Hidden "..n)
			local lbl = newText(container, tag, UDim2.fromOffset(colX[li], padding*0.5), Vector2.new(0.5,0), (li==1 and palette[1]) or (li==L and palette[3]) or palette[2])
			labelFrames[li] = lbl
		end

		-- nodes
		local nf = newFrame(container, ("Nodes_%d"):format(li), UDim2.fromOffset(1,1), UDim2.fromOffset(0,0), Vector2.new(0,0), nil, 1)
		nf.ZIndex = 2
		nodeFolders[li] = nf
		local posList = table.create(count)
		for idx=1,count do
			local j = picks[idx]
			local y = topY + (idx-1) * (2*nodeR + rowMinGap) + nodeR
			posList[idx] = Vector2.new(colX[li], y)
			local node = newCircle(nf, posList[idx], nodeR, Color3.fromRGB(210,210,210))
			node.Name = ("n_%d"):format(j)
		end
		nodePos[li] = posList
	end

	local Ws = collectWeightMats(params)
	local edgeSpecs = {} :: {
		{
			W: Tensor, mode: string,
			inPick:{number}, outPick:{number},
			pointsIn:{Vector2}, pointsOut:{Vector2},
			lines:{Frame}, emaMax:number
		}
	}

	local linkCount = math.min(#Ws, L - 1)
	for li = 1, linkCount do
		local W = Ws[li]

		local fullInPick = nodePicks[li] or {}
		local fullOutPick = nodePicks[li+1] or {}

		local targetSide = math.floor(math.sqrt(maxLinesPerEdge))
		local strideIn = math.max(1, math.floor(#fullInPick / math.max(1, targetSide)))
		local strideOut = math.max(1, math.floor(#fullOutPick / math.max(1, targetSide)))

		local selIn, selOut = {}, {}
		for k=1,#fullInPick, strideIn do table.insert(selIn, fullInPick[k]) end
		for k=1,#fullOutPick, strideOut do table.insert(selOut, fullOutPick[k]) end

		local lineFolder = newFrame(container, ("Lines_%d"):format(li), UDim2.fromOffset(1,1), UDim2.fromOffset(0,0), Vector2.new(0,0), nil, 1)
		local _pairs = {}
		for oi=1,#selOut do
			for ij=1,#selIn do
				if #_pairs >= maxLinesPerEdge then break end
				local p1 = posByIndex(nodePicks, nodePos, sizes, li, selIn[ij])
				local p2 = posByIndex(nodePicks, nodePos, sizes, li + 1, selOut[oi])
				local f = newLine(lineFolder, p1, p2, lineMin, Color3.fromRGB(180,180,180))
				table.insert(_pairs, { iOut = selOut[oi], jIn = selIn[ij], line = f })
			end
			if #_pairs >= maxLinesPerEdge then break end
		end

		table.insert(edgeSpecs, {
			W = W,
			pairs = _pairs,
			emaMax = 1.0,
		})
	end

	local function getValueField(t: Tensor, idx: number, useGrad: boolean): number
		if useGrad and t._grad and t._grad._storage then
			return math.abs(t._grad._storage[idx] or 0)
		end
		return t._storage[idx] or 0
	end

	local autoConn: RBXScriptConnection? = nil

	local handle: RenderHandle
	handle = {
		container = container,

		setMode = function(self, m: string)
			if m == "grads" or m == "weights" then currentMode = m end
		end,

		update = function(self)
			for _,spec in ipairs(edgeSpecs) do
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

					local thick = lineMin + (lineMax - lineMin) * mag
					local color = (useGrad and Color3.fromRGB(255,220,90))
						or ((val >= 0) and Color3.fromRGB(90,180,255) or Color3.fromRGB(255,120,120))

					local f = pr.line
					f.Size = UDim2.fromOffset(f.Size.X.Offset, thick)
					f.BackgroundColor3 = color
				end
			end

			if opt.getActivations then
				local acts = opt.getActivations(model)
				if type(acts) == "table" then
					for li=1,#sizes do
						local layerActs = acts[li]
						if layerActs then
							local nf = nodeFolders[li]
							local picks = nodePicks[li]
							for idx=1,#picks do
								local j = picks[idx]
								local node = nf:FindFirstChild(("n_%d"):format(j)) :: Frame
								if node then
									local a = layerActs[j] or 0
									local s = clamp(math.abs(a), 0, 1)
									node.BackgroundColor3 = (a >= 0) and Color3.fromRGB(150, 220, 255) or Color3.fromRGB(255, 160, 160)
									node.BackgroundTransparency = 1 - (0.15 + 0.75 * s)
									local r = nodeR + 3*s
									node.Size = UDim2.fromOffset(r*2, r*2)
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
				local acc = 0.0
				autoConn = RunService.PreSimulation:Connect(function(dt)
					acc += dt
					if acc >= updateEvery then
						acc = 0.0
						self:update()
					end
				end)
			end
		end,

		destroy = function(self)
			if autoConn then autoConn:Disconnect(); autoConn = nil end
			self.container:Destroy()
		end,
	}

	handle:update()
	handle:setAutoUpdate(true)
	return handle
end

return Visual2D
