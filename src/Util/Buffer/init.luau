--!native
--!native
--!optimize 2
--@Eternity_Devs
--from Warp lib & extended
local Buffer = {}
Buffer.__index = Buffer

local Dedicated = require("@self/Dedicated")

local function readValue(b: buffer, position: number, ref: { any }?): (any, number)
	local typeByte = buffer.readi8(b, position)
	position += 1
	if typeByte == 0 then -- nil
		return nil, position
	elseif typeByte == -1 then -- Instance
		if type(ref) ~= "table" then
			return nil, position + 1 -- still consume the index (compat if kept u8)
		end
		local idx = buffer.readu16(b, position)
		local value = ref[idx]
		if typeof(value) == "Instance" then
			return value, position + 2
		end
		return nil, position + 2
	elseif typeByte == -2 then -- array
		local length = buffer.readu16(b, position)
		position += 2
		local array = {}
		for _ = 1, length do
			local value
			value, position = readValue(b, position, ref)
			table.insert(array, value)
		end
		return array, position
	elseif typeByte == -3 then -- dictionary
		local length = buffer.readu16(b, position)
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
		local length = buffer.readu8(b, position)
		local enumName = buffer.readstring(b, position + 1, length)
		local enumVal = buffer.readu16(b, position + 1 + length)
		return Enum[enumName]:FromValue(enumVal), position + 3 + length
	elseif typeByte == -5 then -- BrickColor
		local value = buffer.readi16(b, position)
		return BrickColor.new(value), position + 2
	elseif typeByte == -6 then -- Enum
		local length = buffer.readu8(b, position)
		local enumName = buffer.readstring(b, position + 1, length)
		return Enum[enumName], position + 1 + length
	elseif typeByte == 1 then -- int u8
		local value = buffer.readu8(b, position)
		return value, position + 1
	elseif typeByte == 2 then -- int i16
		local value = buffer.readi16(b, position)
		return value, position + 2
	elseif typeByte == 3 then -- int i32
		local value = buffer.readi32(b, position)
		return value, position + 4
	elseif typeByte == 4 then -- f64
		local value = buffer.readf64(b, position)
		return value, position + 8
	elseif typeByte == 5 then
		return true, position
	elseif typeByte == 6 then
		return false, position
	elseif typeByte == 7 then -- string u8
		local length = buffer.readu8(b, position)
		local value = buffer.readstring(b, position + 1, length)
		return value, position + length + 1
	elseif typeByte == 8 then -- string u16
		local length = buffer.readu16(b, position)
		local value = buffer.readstring(b, position + 2, length)
		return value, position + length + 2
	elseif typeByte == 9 then -- string i32
		local length = buffer.readi32(b, position)
		local value = buffer.readstring(b, position + 4, length)
		return value, position + length + 4
	elseif typeByte == 10 then -- Vector3
		local x = buffer.readf32(b, position)
		local y = buffer.readf32(b, position + 4)
		local z = buffer.readf32(b, position + 8)
		return Vector3.new(x, y, z), position + 12
	elseif typeByte == 11 then -- Vector2
		local x = buffer.readf32(b, position)
		local y = buffer.readf32(b, position + 4)
		return Vector2.new(x, y), position + 8
	elseif typeByte == 12 then -- CFrame
		local components = {}
		for i = 1, 12 do
			table.insert(components, buffer.readf32(b, position + (i - 1) * 4))
		end
		return CFrame.new(unpack(components)), position + 48
	elseif typeByte == 13 then -- Color3
		local r = buffer.readu8(b, position)
		local g = buffer.readu8(b, position + 1)
		local channelB = buffer.readu8(b, position + 2)
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
	return buffer.fromstring(s)
end

function Buffer.write(data: any): (buffer, any?)
	local newBuffer = Dedicated()
	newBuffer:pack(data)
	return newBuffer:buildAndRemove()
end

function Buffer.read(b: buffer, ref: { any }?): any?
	local position = 0
	local result = {}
	while position < buffer.len(b) do
		local value
		value, position = readValue(b, position, ref)
		table.insert(result, value)
	end
	ref = nil
	return result
end

function Buffer.writeDenseArray(data: { number }, existing: buffer?): buffer
	local count = #data
	local totalBytes = count * 8
	local buf = existing
	if not buf or buffer.len(buf) ~= totalBytes then
		buf = buffer.create(totalBytes)
	end
	local offset = 0
	for i = 1, count do
		buffer.writef64(buf, offset, data[i])
		offset += 8
	end
	return buf
end

function Buffer.readDenseArray(buf: buffer, out: { number }?): { number }
	local count = math.floor(buffer.len(buf) / 8)
	local dest = out
	if not dest or #dest ~= count then
		dest = table.create(count)
	end
	local offset = 0
	for i = 1, count do
		dest[i] = buffer.readf64(buf, offset)
		offset += 8
	end
	return dest
end

function Buffer.copyDenseToColumn(
	buf: buffer,
	target: { number },
	column: number,
	batch: number,
	length: number?
)
	local elements = length or math.floor(buffer.len(buf) / 8)
	local offset = 0
	local idx = column
	for _ = 1, elements do
		target[idx] = buffer.readf64(buf, offset)
		offset += 8
		idx += batch
	end
end

return Buffer :: typeof(Buffer)
