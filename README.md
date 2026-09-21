# AI Research Foundations Multilingual Tokenization Challenge

Build **one tokenizer for six languages** under a fixed **10,000-token vocabulary budget**.

Your tokenizer must represent:

**English · French · Hausa · Swahili · Yoruba · Amharic**

Your leaderboard score is based on **Hausa, Swahili, Yoruba and Amharic**. English and French are included as a guardrail, so you cannot simply optimise for the four target languages and ignore the others.

**Lower score = better.**

|                 |                                      |
| --------------- | ------------------------------------ |
| **Start here**  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aims-ai-research-foundations/airf-multilingual-tokenizer-challenge/blob/main/starter/starter.ipynb) [Open the starter notebook in Colab](https://colab.research.google.com/github/aims-ai-research-foundations/airf-multilingual-tokenizer-challenge/blob/main/starter/starter.ipynb) |
| **Leaderboard** | [View current standings](https://airf.aims.ac.za/community/africa-multilingual-tokenizer-challenge/leaderboard/), also in [LEADERBOARD.md](LEADERBOARD.md) |
| **Submission**  | [Read the submission guide](CONTRIBUTING.md) |

## The Challenge

Design a tokenizer that represents the competition data using as few tokens as possible while remaining **lossless, multilingual and computationally efficient**.

You have:

* **10,000 tokens** maximum
* **240,000 training examples**
* **24,000 validation examples**
* **6 languages**
* **No external data or pretrained tokenizers**

The test set is hidden and is used for the official leaderboard.

## Dataset

| Language | Code |  Train | Validation |
| -------- | ---- | -----: | ---------: |
| English  | `en` | 40,000 |      4,000 |
| French   | `fr` | 40,000 |      4,000 |
| Hausa    | `ha` | 40,000 |      4,000 |
| Swahili  | `sw` | 40,000 |      4,000 |
| Yoruba   | `yo` | 40,000 |      4,000 |
| Amharic  | `am` | 40,000 |      4,000 |

The text comes from a pinned Wikipedia snapshot, is normalised to Unicode NFC, and globally deduplicated. The public train and validation sets are available on the Hugging Face Hub at [`Similoluwa/african-multilingual-tokenizer-challenge`](https://huggingface.co/datasets/Similoluwa/african-multilingual-tokenizer-challenge), pinned to revision `v1.0.0`.

```python
from datasets import load_dataset

train = load_dataset("Similoluwa/african-multilingual-tokenizer-challenge",
                     split="train", revision="v1.0.0")
```

## Rules

Your submission must:

* support all six languages;
* contain **no more than 10,000 tokens**;
* be trained from the provided competition training data;
* use no pretrained tokenizer, external corpus, vocabulary, word list, embedding or API;
* run offline;
* restore the text it encodes;
* finish within **5× the character-level baseline runtime**;
* pass the official submission checker.

You may use the Hugging Face [`tokenizers`](https://github.com/huggingface/tokenizers) library and any tokenizer algorithm you can implement or configure yourself.

**You must also submit the notebook used to build your tokenizer before the deadline. Failure to submit it results in disqualification.**

## Evaluation

The objective is simple:

> **Minimise token usage without losing information.**

### Token Fertility

For each language:

$$
F_l =
\frac{\text{number of tokens}}
{\text{number of whitespace-separated words}}
$$

Lower is better.

### Unknown Tokens

Unknown tokens indicate that the tokenizer cannot represent part of the text:

$$
U_l =
\frac{\text{number of } \texttt{[UNK]} \text{ tokens}}
{\text{number of words}}
$$

The language score is:

$$
S_l = F_l + 100U_l
$$

A 1% `[UNK]` rate therefore adds **1.00** to the language score.

### Final Score

For the four target languages:

$$
T=\{ha,sw,yo,am\}
$$

$$
S_{\text{target}}
=
\frac{1}{4}
\sum_{l\in T} S_l
$$

Lower is better.

### English and French Guardrail

English and French are not directly scored. However, each is allowed a fertility of up to **1.15× the average fertility of the four target languages**.

Any excess is added to your score:

$$
P_{en/fr}
=
\sum_{l\in\{en,fr\}}
\max(0,F_l-1.15\bar F)
$$

### Reconstruction

A tokenizer should preserve the text it encodes. We therefore check whether `decode(encode(text))` exactly reconstructs the original text.

The reconstruction rate is the proportion of evaluation examples reconstructed exactly:

$$
\text{reconstruction}
=
\frac{\text{rows where decode(encode(text)) exactly matches text}}
{\text{total rows}}
$$

Any reconstruction failure adds a penalty:

$$
P_{\text{reconstruction}} = 3(1-\text{reconstruction})
$$

A tokenizer that reconstructs every example exactly receives no penalty.

The penalty discourages approaches that artificially reduce token counts by modifying or deleting information from the input.

Unicode NFC, special tokens, and leading/trailing whitespace are ignored.

### Complete Score

$$
\boxed{
\text{Score}
=
S_{\text{target}}
+
P_{en/fr}
+
P_{\text{reconstruction}}
}
$$

**Lower is better.**

## Baselines

| Baseline        | Score | Key trade-off                             |
| --------------- | ----: | ----------------------------------------- |
| Word-level      | 36.99 | Very low fertility, but high `[UNK]` rate |
| Character-level |  5.68 | Full coverage, but high fertility         |

A good subword tokenizer should beat both. Both are in [starter/baselines/](starter/baselines/).

Try BPE, WordPiece, Unigram, byte-level approaches, custom pre-tokenisation, language-aware sampling, or other strategies.

## Submission

1. Build your tokenizer.
2. Export it as `tokenizer.json`.
3. Run the submission checker.
4. Fork the repository and create a branch named `submission`.
5. Add `tokenizer.json` and `metadata.yml` under `submissions/<team-name>/`.
6. Push the branch and open a pull request.
7. Add your training notebook as `notebook.ipynb` before the deadline.

Your **latest tokenizer on the `submission` branch** is the one that will be evaluated.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the complete submission format.

## Submission Checklist

* [ ] Supports all six languages
* [ ] Vocabulary ≤ 10,000 tokens
* [ ] Uses only the supplied training data
* [ ] No pretrained tokenizer or external resources
* [ ] No external APIs or network access
* [ ] `tokenizer.json` loads correctly
* [ ] Reconstruction checked
* [ ] `[UNK]` rate checked
* [ ] English/French guardrail checked
* [ ] Submission checker passes
* [ ] `metadata.yml` included
* [ ] `notebook.ipynb` included before the deadline

## Repository

```text
starter/
├── starter.ipynb
├── utils.py
├── submission_checker.py
└── baselines/

submissions/
competition/
scripts/
tests/

LEADERBOARD.md
leaderboard.json
```

The starter notebook runs on a free Colab CPU and installs its own dependencies.

## References

The challenge builds on the **AI Research Foundations** learning path:

* [Course 01: Build Your Own Small Language Model](https://www.skills.google/course_templates/1341)
* [Course 02: Represent Your Language Data](https://www.skills.google/course_templates/1452)

[AI Research Foundations Learning Path](https://www.skills.google/paths/3135)

## License

Code is released under the [MIT License](LICENSE). Dataset text is derived from Wikipedia and remains subject to its original licensing terms. See the [dataset card](https://huggingface.co/datasets/Similoluwa/african-multilingual-tokenizer-challenge) for attribution details.
