--!native
--!optimize 2
local ByteLevel = {}
ByteLevel.__index = ByteLevel

local Types = require("../Types")

function ByteLevel.new(trim_offsets: boolean?)
	local self = setmetatable({}, ByteLevel)
	self.trim_offsets = trim_offsets == nil and true or trim_offsets
	return self
end

function ByteLevel:process(encoding: Types.Encoding, pair_encoding: Types.Encoding?): Types.Encoding
	if pair_encoding then
		local new_ids = {}
		local new_tokens = {}
		local new_type_ids = {}
		local new_attention_mask = {}
		local new_special_tokens_mask = {}
		local new_offsets = {}
		local function add(enc, type_id)
			for i = 1, #enc.ids do
				table.insert(new_ids, enc.ids[i])
				table.insert(new_tokens, enc.tokens[i])
				table.insert(new_type_ids, type_id)
				table.insert(new_attention_mask, enc.attention_mask[i])
				table.insert(new_special_tokens_mask, enc.special_tokens_mask[i])
				table.insert(new_offsets, enc.offsets[i])
			end
		end
		add(encoding, 0)
		add(pair_encoding, 1)
		return {
			ids = new_ids,
			tokens = new_tokens,
			offsets = new_offsets,
			attention_mask = new_attention_mask,
			special_tokens_mask = new_special_tokens_mask,
			type_ids = new_type_ids,
		}
	end
	return encoding
end

return ByteLevel
