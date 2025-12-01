--!native
--!optimize 2
local Types = require("../Types")

local TemplateProcessing = {}
TemplateProcessing.__index = TemplateProcessing

export type Template = { string | { id: string, type_id: number } }

function TemplateProcessing.new(single: Template, pair: Template, special_tokens: { [string]: number })
	local self = setmetatable({}, TemplateProcessing)
	self.single = single
	self.pair = pair
	self.special_tokens = special_tokens
	return self
end

function TemplateProcessing:process(encoding: Types.Encoding, pair_encoding: Types.Encoding?): Types.Encoding
	local template = if pair_encoding then self.pair else self.single

	local new_ids = {}
	local new_tokens = {}
	local new_type_ids = {}
	local new_attention_mask = {}
	local new_special_tokens_mask = {}
	local new_offsets = {}

	local function add_seq(enc: Types.Encoding, type_id: number)
		for i = 1, #enc.ids do
			table.insert(new_ids, enc.ids[i])
			table.insert(new_tokens, enc.tokens[i])
			table.insert(new_type_ids, type_id)
			table.insert(new_attention_mask, 1)
			table.insert(new_special_tokens_mask, 0)
			table.insert(new_offsets, enc.offsets[i])
		end
	end
	local function add_special(token: string, type_id: number)
		local id = self.special_tokens[token]
		if not id then
			warn("Special token not found in map: " .. token)
			id = 0
		end
		table.insert(new_ids, id)
		table.insert(new_tokens, token)
		table.insert(new_type_ids, type_id)
		table.insert(new_attention_mask, 1)
		table.insert(new_special_tokens_mask, 1)
		table.insert(new_offsets, { 0, 0 })
	end
	for _, item in ipairs(template) do
		if type(item) == "string" then
			if item == "$A" then
				add_seq(encoding, 0)
			elseif item == "$B" then
				if pair_encoding then
					add_seq(pair_encoding, 1)
				end
			else
				add_special(item, 0)
			end
		elseif type(item) == "table" then
			add_special(item.id, item.type_id)
		end
	end
	return {
		ids = new_ids,
		tokens = new_tokens,
		offsets = new_offsets,
		attention_mask = new_attention_mask,
		special_tokens_mask = new_special_tokens_mask,
		type_ids = new_type_ids,
	}
end

return TemplateProcessing
