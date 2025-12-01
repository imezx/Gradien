--!optimize 2
local BPEDecoder = {}
BPEDecoder.__index = BPEDecoder

local magic = {
	["^"] = "%^",
	["$"] = "%$",
	["("] = "%(",
	[")"] = "%)",
	["%"] = "%%",
	["."] = "%.",
	["["] = "%[",
	["]"] = "%]",
	["*"] = "%*",
	["+"] = "%+",
	["-"] = "%-",
	["?"] = "%?",
}

function BPEDecoder.new(suffix: string?, suffix_replacement: string?)
	local self = setmetatable({}, BPEDecoder)
	self.suffix = suffix or "</w>"
	self.suffix_replacement = suffix_replacement or " "
	return self
end

function BPEDecoder:decode(tokens: { string })
	local text = table.concat(tokens, "")
	local pattern = string.gsub(self.suffix, ".", magic)
	text = string.gsub(text, pattern, self.suffix_replacement)
	return text:match("^%s*(.-)%s*$") or ""
end

return BPEDecoder
