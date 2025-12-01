--!optimize 2
local Sequence = {}
Sequence.__index = Sequence

local Types = require("../Types")

function Sequence.new(normalizers: { Types.Normalizer })
	local self = setmetatable({}, Sequence)
	self.normalizers = normalizers or {}
	return self
end

function Sequence:normalize(text: string)
	for _, n in ipairs(self.normalizers) do
		text = n:normalize(text)
	end
	return text
end

Sequence.from_dump = Sequence.new

return Sequence
