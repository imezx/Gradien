--!native
--!optimize 2
local Whitespace = {}
Whitespace.__index = Whitespace

function Whitespace.new()
	return setmetatable({}, Whitespace)
end

function Whitespace:pre_tokenize(text: string)
	local splits = {}
	local offsets = {}
	local start_idx = 1
	local len = #text
	while start_idx <= len do
		local s, e = string.find(text, "%S+", start_idx)
		if not s then
			break
		end
		table.insert(splits, string.sub(text, s, e))
		table.insert(offsets, { s, e })
		start_idx = e + 1
	end
	return splits, offsets
end

return Whitespace
