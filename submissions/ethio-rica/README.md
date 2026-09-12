# Byte-fallback BPE with multilingual normalization

**Model.** BPE with a 10,000-entry vocabulary, 256 byte tokens as fallback
(so no `[UNK]` is ever produced), `Whitespace` pre-tokenizer, `ByteFallback`
decoder. Trained only on the competition training split with the scored
languages oversampled (ha/sw/yo ×2, am ×3).

**Normalization** (applied in order; each step was measured on validation):

| step | validation score |
|---|---|
| NFC only | 1.9697 |
| + lowercase | 1.8803 |
| + remove punctuation | 1.6665 |
| + collapse digit runs to `0` | 1.6378 |
| + Geʽez homophone merging (ሐ/ኀ→ሀ, ሠ→ሰ, ዐ→አ, ፀ→ጸ) | 1.6310 |
| + NFD + strip accents | 1.5595 |
| + Geʽez vowel-order collapse (all orders → 6th order) | 1.4408 |


