--!native
--!native
--!optimize 2
--@Eternity_Devs
--from Warp lib
local DedicatedBuffer = {}

local create = buffer.create
local copy = buffer.copy
local writei8 = buffer.writei8
local writei16 = buffer.writei16
local writei32 = buffer.writei32
local writeu8 = buffer.writeu8
local writeu16 = buffer.writeu16
local writeu32 = buffer.writeu32
local writef32 = buffer.writef32
local writef64 = buffer.writef64
local writestring = buffer.writestring

local default: { [string]: number } = {
	point = 0,
	next = 0,
	size = 128,
	bufferSize = 128,
}

local function copyMethod(self: any, offset: number, b: buffer?, src: buffer?, srcOffset: number?, count: number?)
	if not b then
		copy(create(count or default.size), offset, src or self.buffer, srcOffset, count)
	else
		copy(b, offset, src or self.buffer, srcOffset, count)
	end
end

local function alloc(self: any, byte: number)
	local need = self.next + byte
	if need > self.size then
		local newSize = self.size
		repeat
			newSize = math.floor(newSize * 1.25 + 0.5)
		until newSize < need and not (newSize < need) or newSize >= need
		if newSize < need then
			newSize = need
		end

		local newBuffer: buffer = create(newSize)
		copy(newBuffer, 0, self.buffer, 0, self.next)
		self.buffer = newBuffer
		self.size = newSize
	end

	self.point = self.next
	self.next += byte
end

local function build(self: any): buffer
	local p: number = self.next > self.point and self.next or self.point
	local buildBuffer: buffer = create(p)

	copy(buildBuffer, 0, self.buffer, 0, p)
	return buildBuffer
end

local function buildAndRemove(self: any): (buffer, any?)
	local p: number = self.next > self.point and self.next or self.point
	local buildBuffer: buffer = create(p)
	local ref = #self.ref > 0 and table.clone(self.ref) or nil

	copy(buildBuffer, 0, self.buffer, 0, p)

	self:remove()
	return buildBuffer, ref
end

local function wi8(self: any, val: number, allocBytes: number?)
	if not val then
		return
	end
	self:alloc(allocBytes or 1)
	writei8(self.buffer, self.point, val)
end

local function wi16(self: any, val: number, allocBytes: number?)
	if not val then
		return
	end
	self:alloc(allocBytes or 2)
	writei16(self.buffer, self.point, val)
end

local function wi32(self: any, val: number, allocBytes: number?)
	if not val then
		return
	end
	self:alloc(allocBytes or 4)
	writei32(self.buffer, self.point, val)
end

local function wu8(self: any, val: number, allocBytes: number?)
	if not val then
		return
	end
	self:alloc(allocBytes or 1)
	writeu8(self.buffer, self.point, val)
end

local function wu16(self: any, val: number, allocBytes: number?)
	if not val then
		return
	end
	self:alloc(allocBytes or 2)
	writeu16(self.buffer, self.point, val)
end

local function wu32(self: any, val: number, allocBytes: number?)
	if not val then
		return
	end
	self:alloc(allocBytes or 4)
	writeu32(self.buffer, self.point, val)
end

local function wf32(self: any, val: number, allocBytes: number?)
	if not val then
		return
	end
	self:alloc(allocBytes or 4)
	writef32(self.buffer, self.point, val)
end

local function wf64(self: any, val: number, allocBytes: number?)
	if not val then
		return
	end
	self:alloc(allocBytes or 8)
	writef64(self.buffer, self.point, val)
end

local function wstring(self: any, val: string)
	if not val then
		return
	end
	self:alloc(#val)
	writestring(self.buffer, self.point, val)
end

local function wType(self: any, ref: number)
	writeu8(self.buffer, self.point, ref)
	self.point += 1
end

local function wRef(self: any, value: any)
	if not value then
		return
	end
	table.insert(self.ref, value)
	local index = #self.ref
	self:alloc(2)
	writeu16(self.buffer, self.point, index)
end

local function pack(self: any, data: any)
	local dataType: string = typeof(data)
	if dataType == "nil" then
		self:wi8(0)
	elseif dataType == "Instance" then
		self:wi8(-1)
		self:wRef(data)
	elseif dataType == "table" then
		local count = 0
		for _ in data do
			count += 1
		end
		local isArray: boolean = (#data == count)
		if isArray then
			self:wi8(-2)
			self:wu16(count)
			for i = 1, count do
				self:pack(data[i])
			end
		else
			self:wi8(-3)
			self:wu16(count)
			for k, v in data do
				self:pack(k)
				self:pack(v)
			end
		end
	elseif dataType == "EnumItem" then
		self:wi8(-4)
		local enumName: string = `{data.EnumType}`
		self:wi8(#enumName)
		self:wstring(enumName)
		self:wu16(data.Value)
	elseif dataType == "BrickColor" then
		self:wi8(-5)
		self:wi16(data.Number)
	elseif dataType == "Enum" then
		self:wi8(-6)
		local enumName: string = data.Name
		self:wi8(#enumName)
		self:wstring(enumName)
	elseif dataType == "number" then
		if math.floor(data) == data then
			if data >= 0 and data <= 255 then
				self:wi8(1)
				self:wu8(data)
			elseif data >= -32768 and data <= 32767 then
				self:wi8(2)
				self:wi16(data)
			elseif data >= -2147483648 and data <= 2147483647 then
				self:wi8(3)
				self:wi32(data)
			else
				self:wi8(4)
				self:wf64(data)
			end
		else
			self:wi8(4)
			self:wf64(data)
		end
	elseif dataType == "boolean" then
		self:wi8(data and 5 or 6)
	elseif dataType == "string" then
		local length = #data
		if length <= 255 then
			self:wi8(7)
			self:wu8(length)
		elseif length <= 65535 then
			self:wi8(8)
			self:wu16(length)
		else
			self:wi8(9)
			self:wi32(length)
		end
		self:wstring(data)
	elseif dataType == "Vector3" then
		self:wi8(10)
		self:wf32(data.X)
		self:wf32(data.Y)
		self:wf32(data.Z)
	elseif dataType == "Vector2" then
		self:wi8(11)
		self:wf32(data.X)
		self:wf32(data.Y)
	elseif dataType == "CFrame" then
		self:wi8(12)
		for _, v in { data:GetComponents() } do
			self:wf32(v)
		end
	elseif dataType == "Color3" then
		self:wi8(13)
		local r = math.clamp(math.floor(data.R * 255 + 0.5), 0, 255)
		local g = math.clamp(math.floor(data.G * 255 + 0.5), 0, 255)
		local b = math.clamp(math.floor(data.B * 255 + 0.5), 0, 255)
		self:wu8(r)
		self:wu8(g)
		self:wu8(b)
	else
		warn(`Unsupported data type: {dataType} value: {data}`)
	end
end

local function flush(self: any)
	self.point = default.point
	self.next = default.next
	self.size = default.size
	self.buffer = create(default.bufferSize)
	table.clear(self.ref)
end

local function remove(self: any)
	self:flush()
	table.clear(self)
end

local methods = {
	copy = copyMethod,
	alloc = alloc,
	build = build,
	buildAndRemove = buildAndRemove,
	wi8 = wi8,
	wi16 = wi16,
	wi32 = wi32,
	wu8 = wu8,
	wu16 = wu16,
	wu32 = wu32,
	wf32 = wf32,
	wf64 = wf64,
	wstring = wstring,
	wType = wType,
	wRef = wRef,
	pack = pack,
	flush = flush,
	remove = remove,
}

for name, fn in methods do
	DedicatedBuffer[name] = fn
end

function DedicatedBuffer.new()
	local obj = {
		point = default.point,
		next = default.next,
		size = default.size,
		buffer = create(default.bufferSize),
		ref = {},
	}
	for name, fn in methods do
		obj[name] = fn
	end
	return obj
end

export type DedicatedType = typeof(DedicatedBuffer.new())

return DedicatedBuffer.new :: typeof(DedicatedBuffer.new)
