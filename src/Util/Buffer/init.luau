--!native
--!optimize 2
--@Eternity_Devs
--from Warp lib & extended
local Buffer = {}
Buffer.__index = Buffer

local Dedicated = require(script.Dedicated)

local tostring = buffer.tostring
local fromstring = buffer.fromstring
local readu8 = buffer.readu8
local readi8 = buffer.readi8
local readu16 = buffer.readu16
local readi16 = buffer.readi16
local readi32 = buffer.readi32
local readf32 = buffer.readf32
local readf64 = buffer.readf64
local readstring = buffer.readstring
local writeu8 = buffer.writeu8
local writei32 = buffer.writei32
local writef32 = buffer.writef32
local writef64 = buffer.writef64
local len = buffer.len
local create = buffer.create

local dtypeInfo = {
	f64 = {
		bytes = 8,
		read = readf64,
		write = writef64,
	},
	f32 = {
		bytes = 4,
		read = readf32,
		write = writef32,
	},
	i32 = {
		bytes = 4,
		read = readi32,
		write = writei32,
	},
	u8 = {
		bytes = 1,
		read = readu8,
		write = writeu8,
	},
}

local function ensureDType(dtype: string?)
	local info = dtypeInfo[dtype or "f64"]
	assert(info, `Buffer: unsupported dtype {dtype}`)
	return info
end

local function readValue(b: buffer, position: number, ref: { any }?): (any, number)
	local typeByte = readi8(b, position)
	position += 1
	if typeByte == 0 then -- nil
		return nil, position
	elseif typeByte == -1 then -- Instance
		if type(ref) ~= "table" then
			return nil, position + 1 -- still consume the index (compat if kept u8)
		end
		local idx = readu16(b, position)
		local value = ref[idx]
		if typeof(value) == "Instance" then
			return value, position + 2
		end
		return nil, position + 2
	elseif typeByte == -2 then -- array
		local length = readu16(b, position)
		position += 2
		local array = {}
		for _ = 1, length do
			local value
			value, position = readValue(b, position, ref)
			table.insert(array, value)
		end
		return array, position
	elseif typeByte == -3 then -- dictionary
		local length = readu16(b, position)
		position += 2
		local dict = {}
		for _ = 1, length do
			local key, value
			key, position = readValue(b, position, ref)
			value, position = readValue(b, position, ref)
			dict[key] = value
		end
		return dict, position
	elseif typeByte == -4 then -- EnumItem
		local length = readu8(b, position)
		local enumName = readstring(b, position + 1, length)
		local enumVal = readu16(b, position + 1 + length)
		return Enum[enumName]:FromValue(enumVal), position + 3 + length
	elseif typeByte == -5 then -- BrickColor
		local value = readi16(b, position)
		return BrickColor.new(value), position + 2
	elseif typeByte == -6 then -- Enum
		local length = readu8(b, position)
		local enumName = readstring(b, position + 1, length)
		return Enum[enumName], position + 1 + length
	elseif typeByte == 1 then -- int u8
		local value = readu8(b, position)
		return value, position + 1
	elseif typeByte == 2 then -- int i16
		local value = readi16(b, position)
		return value, position + 2
	elseif typeByte == 3 then -- int i32
		local value = readi32(b, position)
		return value, position + 4
	elseif typeByte == 4 then -- f64
		local value = readf64(b, position)
		return value, position + 8
	elseif typeByte == 5 then
		return true, position
	elseif typeByte == 6 then
		return false, position
	elseif typeByte == 7 then -- string u8
		local length = readu8(b, position)
		local value = readstring(b, position + 1, length)
		return value, position + length + 1
	elseif typeByte == 8 then -- string u16
		local length = readu16(b, position)
		local value = readstring(b, position + 2, length)
		return value, position + length + 2
	elseif typeByte == 9 then -- string i32
		local length = readi32(b, position)
		local value = readstring(b, position + 4, length)
		return value, position + length + 4
	elseif typeByte == 10 then -- Vector3
		local x = readf32(b, position)
		local y = readf32(b, position + 4)
		local z = readf32(b, position + 8)
		return Vector3.new(x, y, z), position + 12
	elseif typeByte == 11 then -- Vector2
		local x = readf32(b, position)
		local y = readf32(b, position + 4)
		return Vector2.new(x, y), position + 8
	elseif typeByte == 12 then -- CFrame
		local components = {}
		for i = 1, 12 do
			table.insert(components, readf32(b, position + (i - 1) * 4))
		end
		return CFrame.new(unpack(components)), position + 48
	elseif typeByte == 13 then -- Color3
		local r = readu8(b, position)
		local g = readu8(b, position + 1)
		local channelB = readu8(b, position + 2)
		return Color3.fromRGB(r, g, channelB), position + 3
	end
	error(`Unsupported type marker: {typeByte}`)
end

function Buffer.new(): Dedicated.DedicatedType
	return Dedicated()
end

function Buffer.convert(b: buffer): string
	return tostring(b)
end

function Buffer.revert(s: string): buffer
	return fromstring(s)
end

function Buffer.write(data: any): (buffer, any?)
	local newBuffer = Dedicated()
	newBuffer:pack(data)
	return newBuffer:buildAndRemove()
end

function Buffer.read(b: buffer, ref: { any }?): any?
	local position = 0
	local result = {}
	while position < len(b) do
		local value
		value, position = readValue(b, position, ref)
		table.insert(result, value)
	end
	ref = nil
	return result
end

function Buffer.writeDenseArray(data: { number }, dtype: string?, existing: buffer?): buffer
	local info = ensureDType(dtype)
	local count = #data
	local totalBytes = count * info.bytes
	local buf = existing
	if not buf or len(buf) ~= totalBytes then
		buf = create(totalBytes)
	end
	local offset = 0
	for i = 1, count do
		info.write(buf, offset, data[i])
		offset += info.bytes
	end
	return buf
end

function Buffer.readDenseArray(buf: buffer, dtype: string?, out: { number }?): { number }
	local info = ensureDType(dtype)
	local bytes = info.bytes
	local count = math.floor(len(buf) / bytes)
	local dest = out
	if not dest or #dest ~= count then
		dest = table.create(count)
	end
	local offset = 0
	for i = 1, count do
		dest[i] = info.read(buf, offset)
		offset += bytes
	end
	return dest
end

function Buffer.copyDenseToColumn(
	buf: buffer,
	dtype: string?,
	target: { number },
	column: number,
	batch: number,
	length: number?
)
	local info = ensureDType(dtype)
	local bytes = info.bytes
	local elements = length or math.floor(len(buf) / bytes)
	local offset = 0
	local idx = column
	for _ = 1, elements do
		target[idx] = info.read(buf, offset)
		offset += bytes
		idx += batch
	end
end

return Buffer :: typeof(Buffer)
