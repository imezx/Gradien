# Tokenizers

The `Gradien.Tokenizers` module provides a comprehensive toolkit for text tokenization, inspired by the Hugging Face Tokenizers library. It supports Byte-Pair Encoding (BPE), normalization, pre-tokenization, and post-processing.

## Tokenizer Class

The main entry point is the `Tokenizer` class, which coordinates the tokenization pipeline.

### Constructor

::: code-group
```lua [Definition]
Tokenizer.new(model: Model) -> Tokenizer
```
:::

Creates a new Tokenizer with the specified model (e.g., BPE).

### Methods

#### `:encode`
Encodes a text (and optional pair text) into a sequence of tokens/IDs.

::: code-group
```lua [Definition]
(text: string, pair: string?) -> Encoding
```
:::

Returns an `Encoding` object containing:
- `ids`: List of token IDs.
- `tokens`: List of token strings.
- `attention_mask`: Mask identifying valid tokens (1) vs padding (0).
- `special_tokens_mask`: Mask identifying special tokens.
- `type_ids`: Segment IDs (if pair is provided).
- `offsets`: (Currently empty/reserved).

#### `:decode`
Decodes a list of IDs back into a string.

::: code-group
```lua [Definition]
(ids: {number}) -> string
```
:::

#### `:train_from_iterator`
Trains the tokenizer model using an iterator that yields strings.

::: code-group
```lua [Definition]
(iterator: () -> string?, trainer: Trainer) -> ()
```
:::

#### Pipeline Configuration
You can customize the tokenizer pipeline using the following setters:

- `:set_normalizer(n: Normalizer)`
- `:set_pre_tokenizer(pt: PreTokenizer)`
- `:set_post_processor(pp: PostProcessor)`
- `:set_decoder(d: Decoder)`

#### Serialization
Save and load the tokenizer configuration.

::: code-group
```lua [Dump]
:dump() -> table
```
```lua [Load]
Tokenizer.from_dump(data: table) -> Tokenizer
```
:::

## Components

### Models
- **BPE**: Byte-Pair Encoding model.

### Normalizers
- **NFKC**: Unicode normalization form KC.

### Pre-Tokenizers
- **ByteLevel**: Splits on bytes.
- **Whitespace**: Splits on whitespace.

### Decoders
- **BPEDecoder**: Decodes BPE tokens.

### Processors
- **ByteLevel**: Post-processing for byte-level BPE.

## Example Usage

```lua
local Tokenizers = Gradien.Tokenizers
local BPE = Tokenizers.models.BPE
local Tokenizer = Tokenizers.Tokenizer

-- Create a BPE model
local model = BPE.new()
local tokenizer = Tokenizer.new(model)

-- Setup pipeline
tokenizer:set_normalizer(Tokenizers.normalizers.NFKC.new())
tokenizer:set_pre_tokenizer(Tokenizers.pre_tokenizers.ByteLevel.new())
tokenizer:set_decoder(Tokenizers.decoders.BPEDecoder.new())

-- Train (simplified example)
local trainer = Tokenizers.trainers.BpeTrainer.new({
    vocab_size = 1000,
    min_frequency = 2,
    special_tokens = {"<unk>", "<s>", "</s>"}
})

local data = {"Hello world", "Hello universe"}
local idx = 0
local function iterator()
    idx += 1
    return data[idx]
end

tokenizer:train_from_iterator(iterator, trainer)

-- Encode
local encoding = tokenizer:encode("Hello world")
print(encoding.tokens)
-- Output: {"Hello", "Ġworld"} (example)

-- Decode
local text = tokenizer:decode(encoding.ids)
print(text)
-- Output: "Hello world"
```
