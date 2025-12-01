--!native
--!optimize 2
local Sequence = {}
Sequence.__index = Sequence

local Types = require("../Types")

function Sequence.new(pre_tokenizers: { Types.PreTokenizer })
	local self = setmetatable({}, Sequence)
	self.pre_tokenizers = pre_tokenizers or {}
	return self
end

function Sequence:pre_tokenize(text: string)
	local current_tokens = { text }
	local current_offsets = { { 1, #text } }
	for _, pt in ipairs(self.pre_tokenizers) do
		local new_tokens = {}
		local new_offsets = {}
		for i, token in ipairs(current_tokens) do
			local sub_tokens, sub_offsets = pt:pre_tokenize(token)
			for j, sub_token in ipairs(sub_tokens) do
				table.insert(new_tokens, sub_token)
				local start_off, end_off = sub_offsets[j][1], sub_offsets[j][2]
				if start_off == 0 and end_off == 0 then
					table.insert(new_offsets, { 0, 0 })
				else
					local base_offset = current_offsets[i][1]
					table.insert(new_offsets, { base_offset + start_off - 1, base_offset + end_off - 1 })
				end
			end
		end
		current_tokens = new_tokens
		current_offsets = new_offsets
	end
	return current_tokens, current_offsets
end

return Sequence
