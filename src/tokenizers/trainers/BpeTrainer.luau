--!native
--!optimize 2
local BpeTrainer = {}
BpeTrainer.__index = BpeTrainer

local Types = require("../Types")
local BPE = require("../models/BPE")

type BpeTrainerOptions = {
	vocab_size: number?,
	min_frequency: number?,
	special_tokens: { string }?,
	show_progress: boolean?,
	initial_alphabet: { string }?,
	limit_alphabet: number?,
	continuing_subword_prefix: string?,
	end_of_word_suffix: string?,
	pre_tokenizer: Types.PreTokenizer?,
}

function BpeTrainer.new(options: BpeTrainerOptions)
	local self = setmetatable({}, BpeTrainer)
	self.vocab_size = options.vocab_size or 30000
	self.min_frequency = options.min_frequency or 0
	self.special_tokens = options.special_tokens or {}
	self.show_progress = options.show_progress or false
	self.initial_alphabet = options.initial_alphabet or {}
	self.limit_alphabet = options.limit_alphabet
	self.continuing_subword_prefix = options.continuing_subword_prefix
	self.end_of_word_suffix = options.end_of_word_suffix or "</w>"
	self.pre_tokenizer = options.pre_tokenizer
	return self
end

function BpeTrainer:train(iterator: () -> string?): (Types.Model, { string })
	local word_counts: { [string]: number } = {}
	local alphabet: { [string]: number } = {}
	if self.show_progress then
		print("Counting words...")
	end
	for text in iterator do
		local words = {}
		if self.pre_tokenizer then
			local pre_tokenized, _ = self.pre_tokenizer:pre_tokenize(text)
			words = pre_tokenized
		else
			for word in string.gmatch(text, "%S+") do
				table.insert(words, word)
			end
		end
		for _, word in ipairs(words) do
			word_counts[word] = (word_counts[word] or 0) + 1
			for i = 1, #word do
				local char = string.sub(word, i, i)
				alphabet[char] = (alphabet[char] or 0) + 1
			end
		end
	end
	local vocab: { [string]: number } = {}
	local merges: { { string } } = {}
	local next_id = 0
	for _, token in ipairs(self.special_tokens) do
		if not vocab[token] then
			vocab[token] = next_id
			next_id += 1
		end
	end
	for _, char in ipairs(self.initial_alphabet) do
		if not alphabet[char] then
			alphabet[char] = 0
		end
	end
	local sorted_alphabet = {}
	for char in pairs(alphabet) do
		table.insert(sorted_alphabet, char)
	end
	table.sort(sorted_alphabet)
	for _, char in ipairs(sorted_alphabet) do
		if not vocab[char] then
			if next_id >= self.vocab_size then
				break
			end
			vocab[char] = next_id
			next_id += 1
		end
	end
	local splits: { { string } } = {}
	local counts: { number } = {}
	for word, count in pairs(word_counts) do
		local split = {}
		for i = 1, #word do
			table.insert(split, string.sub(word, i, i))
		end
		if self.end_of_word_suffix then
			split[#split] ..= self.end_of_word_suffix
		end
		table.insert(splits, split)
		table.insert(counts, count)
	end
	local initial_symbols = {}
	for _, split in ipairs(splits) do
		for _, symbol in ipairs(split) do
			if not vocab[symbol] then
				initial_symbols[symbol] = true
			end
		end
	end
	local sorted_symbols = {}
	for s in pairs(initial_symbols) do
		table.insert(sorted_symbols, s)
	end
	table.sort(sorted_symbols)
	for _, s in ipairs(sorted_symbols) do
		if next_id < self.vocab_size then
			vocab[s] = next_id
			next_id += 1
		end
	end
	if self.show_progress then
		print("Training BPE...")
	end
	local pair_counts: { [string]: number } = {}
	for i, split in ipairs(splits) do
		local count = counts[i]
		if #split >= 2 then
			for j = 1, #split - 1 do
				local pair = `{split[j]}\0{split[j + 1]}`
				pair_counts[pair] = (pair_counts[pair] or 0) + count
			end
		end
	end
	while next_id < self.vocab_size do
		local best_pair = nil
		local max_freq = -1
		for pair, freq in pairs(pair_counts) do
			if freq > max_freq then
				max_freq = freq
				best_pair = pair
			elseif freq == max_freq then
				if best_pair and pair < best_pair then
					best_pair = pair
				end
			end
		end
		if not best_pair or max_freq < self.min_frequency then
			break
		end
		local p1, p2 = best_pair:match("^(.*)%z(.*)$")
		local merged_token = `{p1}{p2}`
		vocab[merged_token] = next_id
		next_id += 1
		table.insert(merges, { p1, p2 })
		pair_counts[best_pair] = nil
		for i, split in ipairs(splits) do
			if #split >= 2 then
				local new_split = {}
				local j = 1
				local changed = false
				while j <= #split do
					if j < #split and split[j] == p1 and split[j + 1] == p2 then
						local count = counts[i]
						if j > 1 then
							local left = new_split[#new_split]
							local pair_left = `{left}\0{p1}`
							if pair_left ~= best_pair then
								pair_counts[pair_left] = (pair_counts[pair_left] or 0) - count
								if pair_counts[pair_left] <= 0 then
									pair_counts[pair_left] = nil
								end
							end
							local new_pair_left = `{left}\0{merged_token}`
							pair_counts[new_pair_left] = (pair_counts[new_pair_left] or 0) + count
						end
						if j + 1 < #split then
							local right = split[j + 2]
							local pair_right = `{p2}\0{right}`
							if pair_right ~= best_pair then
								pair_counts[pair_right] = (pair_counts[pair_right] or 0) - count
								if pair_counts[pair_right] <= 0 then
									pair_counts[pair_right] = nil
								end
							end
							local new_pair_right = `{merged_token}\0{right}`
							pair_counts[new_pair_right] = (pair_counts[new_pair_right] or 0) + count
						end
						table.insert(new_split, merged_token)
						j += 2
						changed = true
					else
						table.insert(new_split, split[j])
						j += 1
					end
				end
				if changed then
					splits[i] = new_split
				end
			end
		end
		if self.show_progress and next_id % 100 == 0 then
			print("Vocab size: " .. next_id)
		end
	end
	return BPE.new(vocab, merges, nil, self.end_of_word_suffix), {}
end

return BpeTrainer
