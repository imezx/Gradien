--!native
--!optimize 2
local BPE = {}
BPE.__index = BPE

function BPE.new(vocab: { [string]: number }?, merges: { { string } }?, unk_token: string?, suffix: string?)
	local self = setmetatable({}, BPE)
	self.vocab = vocab or {}
	self.merges = merges or {}
	self.unk_token = unk_token or "[UNK]"
	self.suffix = suffix or "</w>"
	self.ids_to_tokens = {}
	for k, v in pairs(self.vocab) do
		self.ids_to_tokens[v] = k
	end
	self.merge_ranks = {}
	if merges then
		for i, pair in ipairs(merges) do
			local key = `{pair[1]} {pair[2]}`
			self.merge_ranks[key] = i
		end
	end
	self.ignore_merges = false
	return self
end

function BPE:set_ignore_merges(ignore: boolean)
	self.ignore_merges = ignore
end

function BPE:tokenize(sequences: { string })
	local tokens = {}
	for _, word in ipairs(sequences) do
		local word_tokens = self:_tokenize_word(word)
		for _, t in ipairs(word_tokens) do
			table.insert(tokens, t)
		end
	end
	return tokens
end

function BPE:_tokenize_word(word: string)
	local splits = {}
	local len = #word
	if len == 0 then
		return {}
	end
	for i = 1, len - 1 do
		table.insert(splits, string.sub(word, i, i))
	end
	table.insert(splits, string.sub(word, len, len) .. self.suffix)
	if self.ignore_merges then
		for i = 1, #splits do
			if not self.vocab[splits[i]] then
				splits[i] = self.unk_token
			end
		end
		return splits
	end
	while #splits > 1 do
		local min_rank = math.huge
		local best_pair_idx = -1
		for i = 1, #splits - 1 do
			local pair = `{splits[i]} {splits[i + 1]}`
			local rank = self.merge_ranks[pair]
			if rank and rank < min_rank then
				min_rank = rank
				best_pair_idx = i
			end
		end
		if best_pair_idx == -1 then
			break
		end
		local p1 = splits[best_pair_idx]
		local p2 = splits[best_pair_idx + 1]
		table.remove(splits, best_pair_idx + 1)
		splits[best_pair_idx] = `{p1}{p2}`
	end
	for i = 1, #splits do
		if not self.vocab[splits[i]] then
			splits[i] = self.unk_token
		end
	end
	return splits
end

function BPE:token_to_id(token: string)
	return self.vocab[token] or self.vocab[self.unk_token]
end

function BPE:id_to_token(id: number)
	return self.ids_to_tokens[id]
end

function BPE:get_vocab()
	return self.vocab
end

function BPE:get_vocab_size()
	local count = 0
	for _ in self.vocab do
		count += 1
	end
	return count
end

function BPE:dump()
	return {
		type = "BPE",
		vocab = self.vocab,
		merges = self.merges,
		unk_token = self.unk_token,
		suffix = self.suffix,
		ignore_merges = self.ignore_merges,
	}
end

function BPE.from_dump(data)
	local self = BPE.new(data.vocab, data.merges, data.unk_token, data.suffix)
	if data.ignore_merges then
		self:set_ignore_merges(data.ignore_merges)
	end
	return self
end

return BPE
