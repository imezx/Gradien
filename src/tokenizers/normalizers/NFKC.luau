--!optimize 2
local NFKC = {}
NFKC.__index = NFKC

function NFKC.new()
	local self = setmetatable({}, NFKC)
	return self
end

function NFKC:normalize(text: string)
	text = string.gsub(text, "\194\160", " ")
	return text
end

function NFKC:dump()
	return {
		type = "NFKC",
	}
end

NFKC.from_dump = NFKC.new

return NFKC
