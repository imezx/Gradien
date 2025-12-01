--!strict
export type Encoding = {
	ids: { number },
	tokens: { string },
	offsets: { { number } },
	attention_mask: { number },
	special_tokens_mask: { number },
	type_ids: { number },
}

export type Model = {
	tokenize: (self: Model, sequences: { string }) -> { any },
	token_to_id: (self: Model, token: string) -> number?,
	id_to_token: (self: Model, id: number) -> string?,
	get_vocab: (self: Model) -> { [string]: number },
	get_vocab_size: (self: Model) -> number,
	save: (self: Model, folder: string, prefix: string?) -> { string },
}

export type PreTokenizer = {
	pre_tokenize: (self: PreTokenizer, text: string) -> ({ string }, { { number } }),
}

export type Decoder = {
	decode: (self: Decoder, tokens: { string }) -> string,
}

export type PostProcessor = {
	process: (self: PostProcessor, encoding: Encoding, pair_encoding: Encoding?) -> Encoding,
}

export type Normalizer = {
	normalize: (self: Normalizer, text: string) -> string,
}

export type Trainer = {
	train: (self: Trainer, iterator: () -> string?) -> (Model, { string }),
}

return nil
