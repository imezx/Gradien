--!native
--!optimize 2
local Types = require("./Types")
local NFKC = require("./normalizers/NFKC")
local ByteLevel = require("./pre_tokenizers/ByteLevel")
local ByteLevelProcessor = require("./processors/ByteLevel")
local BPE = require("./models/BPE")
local BPEDecoder = require("./decoders/BPEDecoder")

local Tokenizer = {}
Tokenizer.__index = Tokenizer

function Tokenizer.new(model: Types.Model)
	local self = setmetatable({}, Tokenizer)
	self.model = model
	self.normalizer = nil
	self.pre_tokenizer = nil
	self.decoder = nil
	self.post_processor = nil
	return self
end

function Tokenizer:encode(text: string, pair: string?): Types.Encoding
	if self.normalizer then
		text = self.normalizer:normalize(text)
		if pair then
			pair = self.normalizer:normalize(pair)
		end
	end
	local words = { text }
	local pair_words = if pair then { pair } else nil
	if self.pre_tokenizer then
		words, _ = self.pre_tokenizer:pre_tokenize(text)
		if pair and pair_words then
			pair_words, _ = self.pre_tokenizer:pre_tokenize(pair)
		end
	end
	local function run_model(seq_words)
		local tokens = self.model:tokenize(seq_words)
		local ids = {}
		for _, t in ipairs(tokens) do
			local id = self.model:token_to_id(t)
			table.insert(ids, id or 0)
		end
		return {
			ids = ids,
			tokens = tokens,
			offsets = {},
			attention_mask = table.create(#ids, 1),
			special_tokens_mask = table.create(#ids, 0),
			type_ids = table.create(#ids, 0),
		}
	end
	local encoding = run_model(words)
	local pair_encoding = if pair_words then run_model(pair_words) else nil
	if self.post_processor then
		encoding = self.post_processor:process(encoding, pair_encoding)
	elseif pair_encoding then
		for i = 1, #pair_encoding.ids do
			table.insert(encoding.ids, pair_encoding.ids[i])
			table.insert(encoding.tokens, pair_encoding.tokens[i])
			table.insert(encoding.attention_mask, 1)
			table.insert(encoding.special_tokens_mask, 0)
			table.insert(encoding.type_ids, 1)
		end
	end
	return encoding
end

function Tokenizer:decode(ids: { number }): string
	local tokens = {}
	for _, id in ipairs(ids) do
		local t = self.model:id_to_token(id)
		if t then
			table.insert(tokens, t)
		end
	end
	if self.decoder then
		return self.decoder:decode(tokens)
	end
	return table.concat(tokens, " ")
end

function Tokenizer:set_pre_tokenizer(pt: Types.PreTokenizer)
	self.pre_tokenizer = pt
end

function Tokenizer:set_post_processor(pp: Types.PostProcessor)
	self.post_processor = pp
end

function Tokenizer:set_decoder(d: Types.Decoder)
	self.decoder = d
end

function Tokenizer:set_normalizer(n: Types.Normalizer)
	self.normalizer = n
end

function Tokenizer:train_from_iterator(iterator: () -> string?, trainer: Types.Trainer)
	local incoming_iterator = iterator
	if self.normalizer then
		local previous_iterator = iterator
		local function wrapped_iterator()
			local text = previous_iterator()
			if text then
				return self.normalizer:normalize(text)
			end
			return nil
		end
		incoming_iterator = wrapped_iterator
	end
	local old_pre_tok = nil
	if (trainer :: any).pre_tokenizer ~= nil then
		old_pre_tok = (trainer :: any).pre_tokenizer
	end
	if self.pre_tokenizer then
		(trainer :: any).pre_tokenizer = self.pre_tokenizer
	end
	local model = trainer:train(incoming_iterator)
	if self.pre_tokenizer then
		(trainer :: any).pre_tokenizer = old_pre_tok
	end
	self.model = model
end

function Tokenizer:dump()
	local dump = {
		model = self.model:dump(),
		normalizer = nil,
		pre_tokenizer = nil,
		post_processor = nil,
		decoder = nil,
	}
	if self.normalizer and self.normalizer.dump then
		dump.normalizer = self.normalizer:dump()
	elseif self.normalizer then
		if getmetatable(self.normalizer) == NFKC then
			dump.normalizer = { type = "NFKC" }
		end
	end
	if self.pre_tokenizer and self.pre_tokenizer.dump then
		dump.pre_tokenizer = self.pre_tokenizer:dump()
	elseif self.pre_tokenizer then
		if getmetatable(self.pre_tokenizer) == ByteLevel then
			dump.pre_tokenizer = {
				type = "ByteLevel",
				add_prefix_space = self.pre_tokenizer.add_prefix_space,
				use_regex = self.pre_tokenizer.use_regex,
			}
		end
	end
	if self.post_processor and self.post_processor.dump then
		dump.post_processor = self.post_processor:dump()
	elseif self.post_processor then
		if getmetatable(self.post_processor) == ByteLevelProcessor then
			dump.post_processor = {
				type = "ByteLevel",
				trim_offsets = self.post_processor.trim_offsets,
			}
		end
	end
	if self.decoder and self.decoder.dump then
		dump.decoder = self.decoder:dump()
	elseif self.decoder then
		if getmetatable(self.decoder) == BPEDecoder then
			dump.decoder = {
				type = "BPEDecoder",
				suffix = self.decoder.suffix,
				suffix_replacement = self.decoder.suffix_replacement,
			}
		end
	end
	return dump
end

function Tokenizer.from_dump(data)
	local model
	if data.model.type == "BPE" then
		model = BPE.from_dump(data.model)
	else
		error("Unknown model type: " .. tostring(data.model.type))
	end
	local tok = Tokenizer.new(model)
	if data.normalizer then
		if data.normalizer.type == "NFKC" then
			tok:set_normalizer(NFKC.new())
		end
	end
	if data.pre_tokenizer then
		if data.pre_tokenizer.type == "ByteLevel" then
			tok:set_pre_tokenizer(ByteLevel.new(data.pre_tokenizer.add_prefix_space, data.pre_tokenizer.use_regex))
		end
	end
	if data.post_processor then
		if data.post_processor.type == "ByteLevel" then
			tok:set_post_processor(ByteLevelProcessor.new(data.post_processor.trim_offsets))
		end
	end
	if data.decoder then
		if data.decoder.type == "BPEDecoder" then
			tok:set_decoder(BPEDecoder.new(data.decoder.suffix, data.decoder.suffix_replacement))
		end
	end
	return tok
end

return Tokenizer
