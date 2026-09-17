# ethio-rica — lossless byte-level BPE for six languages

## Design

**Pipeline:** NFC → split into units → bytes → BPE merges.

- **Normalizer:** Unicode NFC only. No lowercasing, no character merging, so
  the text is never altered.
- **Split rule (our own):** `Regex(" ?\w+| ?[^\w\s]+|\s+")` — a word together
  with the space before it, a run of punctuation with the space before it, or
  any other whitespace. Spaces travel with the following word so decoding can
  restore them; punctuation is split off so `dunia` and `dunia.` share tokens.
- **Byte mapping:** `ByteLevel(use_regex=False)` rewrites each unit as its
  UTF-8 bytes, giving a fixed 256-symbol alphabet for all six scripts. Every
  input is representable, so there is no `[UNK]` token at all.
- **Model:** BPE, 10,000 entries — the 256 byte symbols plus 9,744 merges
  learned from the competition training split. Merges rebuild multi-byte
  characters first (each Geʽez character is 3 bytes), then syllables,
  affixes, and frequent whole words.
- **Decoder:** `ByteLevel` — bytes back to text. `decode(encode(text))`
  reproduces the input exactly (100% on validation).
- **Language weighting:** target languages oversampled during training
  (ha/sw/yo ×2, am ×3) so they claim more of the merge budget.

## Validation result

| | value |
|---|---|
| Target-language score (avg of ha/sw/yo/am) | X.XXXX |
| `[UNK]` rate | 0 in all six languages |
| Reconstruction | 100% |
| en/fr guardrail excess | 0 |


## Limitations

A word at the very start of a passage (no preceding space) is a different
symbol string from the same word after a space, so a few vocabulary slots
are spent twice. This is the cost of exact reconstruction and is small in
practice.
