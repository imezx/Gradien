--!native
--!optimize 2
local ByteLevel = {}
ByteLevel.__index = ByteLevel

function ByteLevel.new(add_prefix_space: boolean?, use_regex: boolean?)
	local self = setmetatable({}, ByteLevel)
	self.add_prefix_space = add_prefix_space == nil and true or add_prefix_space
	self.use_regex = use_regex == nil and true or use_regex
	return self
end

function ByteLevel:pre_tokenize(text: string)
	local original_text = text
	if self.add_prefix_space and not text:match("^%s") then
		text = ` {text}`
	end
	local offset_shift = if self.add_prefix_space and text ~= original_text then -1 else 0
	local splits = {}
	local offsets = {}
	local current_token = ""
	local last_type = nil
	local token_start = 1
	for i = 1, #text do
		local char = string.sub(text, i, i)
		local char_type = "other"
		if char:match("%s") then
			char_type = "space"
		elseif char:match("%a") then
			char_type = "letter"
		elseif char:match("%d") then
			char_type = "number"
		elseif char:match("%p") then
			char_type = "punct"
		end
		local split = false
		if last_type == nil then
		elseif last_type == "space" and char_type ~= "space" then
			split = false
		elseif last_type ~= "space" and char_type == "space" then
			split = true
		elseif last_type ~= char_type then
			split = true
		end
		if split then
			table.insert(splits, current_token)
			local s = math.max(1, token_start + offset_shift)
			local e = math.max(0, (i - 1) + offset_shift)
			if e < s then
				table.insert(offsets, { 0, 0 })
			else
				table.insert(offsets, { s, e })
			end
			current_token = ""
			token_start = i
		end
		current_token = `{current_token}{char}`
		last_type = char_type
	end
	if #current_token > 0 then
		table.insert(splits, current_token)
		local s = math.max(1, token_start + offset_shift)
		local e = math.max(0, #text + offset_shift)
		if e < s then
			table.insert(offsets, { 0, 0 })
		else
			table.insert(offsets, { s, e })
		end
	end
	return splits, offsets
end

function ByteLevel:alphabet()
	local chars = {}
	for i = 0, 255 do
		table.insert(chars, string.char(i))
	end
	return chars
end

return ByteLevel
