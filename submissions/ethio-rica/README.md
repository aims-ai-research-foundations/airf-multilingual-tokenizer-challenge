# ethio-rica — byte-level BPE without word splitting

A lossless 10,000-entry BPE tokenizer for English, French, Hausa, Swahili,
Yoruba and Amharic. Text is turned into bytes and is **not** cut into words
before training, so one token can cover several words. Rows of the four scored
languages are repeated more often during training so that they receive most of
the vocabulary.

## Pipeline

NFC → UTF-8 bytes → BPE merges

- **Normalizer:** Unicode NFC only. No lowercasing and no character removal,
  so the text is never changed in a way that cannot be undone. NFC gives
  accented letters (common in Yoruba) one single byte form.
- **Pre-tokenizer:** `ByteLevel(add_prefix_space=False, use_regex=False)`.
  Every row becomes its UTF-8 bytes, drawn as 256 printable symbols. With
  `use_regex=False` the row is not split at spaces or punctuation: the whole
  row is one sequence, and the space is an ordinary symbol.
- **Model:** BPE with 10,000 entries: the 256 byte symbols plus 9,744 merges.
  All 256 byte symbols are forced into the vocabulary (`initial_alphabet`), so
  any input can be encoded and there is no `[UNK]` token. There are no special
  tokens.
- **Decoder:** `ByteLevel`. `decode(encode(text))` returns the NFC form of the
  input exactly.

Because merges may cross spaces, frequent phrases become single tokens
(for example `" to "`, `"kwa "`, and long repeated phrases in the Amharic and
Swahili data). The metric is tokens per word, so these tokens lower the score
below what a word-bounded tokenizer can reach.

## Language weighting

The trainer chooses merges by pair frequency, and the 10,000 entries are shared
by six languages. To steer that budget, each language's rows are repeated
according to a weight:

| en | fr | ha | sw | yo | am |
|---|---|---|---|---|---|
| 1.0 | 1.25 | 4.25 | 7.5 | 5.5 | 5.5 |

Fractional weights are handled by a simple running total. Row number `i` of a
language with weight `w` is repeated `int((i + 1) * w) - int(i * w)` times, so
a weight of 7.5 gives 7, 8, 7, 8, … copies. Every training row is used at least
once. The 240,000 training rows become 1,000,000 weighted rows. Identical rows
are counted once with a multiplier by the trainer, so this does not slow
training.

Why these values:

- The score divides each language's tokens by its own word count. Swahili rows
  contain the fewest words, so one saved Swahili token lowers the score more
  than one saved token in the other languages. Swahili therefore gets the
  largest weight.
- English and French are not scored. They only need to stay under the
  guardrail (1.15 × the score), so they get just enough weight for that.

## Results

Validation split (24,000 rows, 4,000 per language), measured with the official
scoring helper in `notebook.ipynb`:

| | ha | sw | yo | am | en | fr |
|---|---|---|---|---|---|---|
| tokens per word | 1.682 | 1.763 | 1.863 | 2.373 | 2.082 | 2.179 |

| | value |
|---|---|
| Score (mean of ha, sw, yo, am) | **1.9202** |
| Guardrail budget (1.15 × score) | 2.208 |
| Guardrail penalty | 0 (en 5.7% under budget, fr 1.3% under) |
| `[UNK]` rate | 0 in all six languages |
| Reconstruction | 100% |
| Vocabulary | 10,000 |

Nightly leaderboard of 2026-09-20 (hidden set): **1.9271**
(ha 1.656, sw 1.751, yo 1.886, am 2.415, en 2.108, fr 2.204).

## Reproducing

Run `notebook.ipynb` from top to bottom (written for Google Colab). It uses
only the official training split, no external data and no pretrained
tokenizer. Library version: `tokenizers==0.22.1`. BPE training is
deterministic, so the same data, weights and library version rebuild the same
merges.

## Limitations

- Tokens that span several words depend on phrases that repeat in this
  dataset. On text from another source the gain will be smaller.
- The space can attach to the word before it or after it, so the same word may
  be split differently depending on its neighbours.
- French is close to the guardrail (1.3% under the budget on validation).
- The method is purely frequency-based and uses no linguistic knowledge of the
  languages.

## Acknowledgement

The final language weights were chosen after reading the public notes of team
`maick-dane-nkou` in this repository, which report that Swahili-heavy
weighting suits this metric. We confirmed the effect on the validation split
(1.9318 → 1.9202) before adopting it.
