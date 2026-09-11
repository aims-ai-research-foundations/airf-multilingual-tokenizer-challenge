# AI Research Foundations Multilingual Tokenization Challenge

## Overview

Tokenization is the first step in how language models turn human language into something they can process, making the design of a tokenizer a fundamental part of building a language model.

But not every tokenizer represents every language equally efficiently.

In this challenge, you will build **one tokenizer for six languages** under a fixed vocabulary budget of **10,000 tokens**.

Your score is measured on four of them: **Hausa, Swahili, Yoruba and Amharic**.
English and French are still part of the data, and you still have to handle them
well, but they are not what you are judged on.

The lower your score, the higher you climb on the leaderboard.

| | |
| --- | --- |
| **Start here** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aims-ai-research-foundations/airf-multilingual-tokenizer-challenge/blob/main/starter/starter.ipynb) &nbsp; [starter/starter.ipynb](starter/starter.ipynb), runs on a free Colab CPU |
| **Current standings** | [LEADERBOARD.md](LEADERBOARD.md), regenerated nightly, and on the [challenge website](https://airf.aims.ac.za/community/africa-multilingual-tokenizer-challenge/leaderboard/) |
| **How to submit** | [CONTRIBUTING.md](CONTRIBUTING.md) |

---

## Dataset

The competition dataset contains text from six languages:

| Language | Code |
| -------- | ---- |
| English  | `en` |
| French   | `fr` |
| Hausa    | `ha` |
| Swahili  | `sw` |
| Yoruba   | `yo` |
| Amharic  | `am` |

The data is divided into three splits:

| Split | Rows per language | Total | Available to you |
| --- | ---: | ---: | :---: |
| Train | 40,000 | 240,000 | yes |
| Validation | 4,000 | 24,000 | yes |
| Test | hidden | hidden | no |

Train and validation are yours to use however you like. The test set stays with
the organizers and produces the official leaderboard score.

Each example contains a language code and a piece of text.

| language | text                                           |
| -------- | ---------------------------------------------- |
| en       | Artificial intelligence is changing the world. |
| yo       | Ẹ̀kọ́ ṣe pàtàkì fún gbogbo ènìyàn.             |
| sw       | Teknolojia inabadilisha dunia.                 |

The public splits are on the Hugging Face Hub at
[`Similoluwa/african-multilingual-tokenizer-challenge`](https://huggingface.co/datasets/Similoluwa/african-multilingual-tokenizer-challenge),
pinned to revision `v1.0.0`. The starter notebook loads them for you:

```python
from datasets import load_dataset

train = load_dataset("Similoluwa/african-multilingual-tokenizer-challenge",
                     split="train", revision="v1.0.0")
```

The text is segmented from a pinned Wikipedia snapshot, normalized to NFC, and
deduplicated globally. Every split is exactly balanced across the six languages.

---

## What You'll Do

* Build **one tokenizer** for all six languages.
* Experiment with different approaches to improve tokenization efficiency.
* Evaluate your ideas on the validation set.
* Work within a fixed **10,000-token vocabulary budget**.
* Submit your best tokenizer and climb the leaderboard.

---

## Rules

Your final tokenizer must:

* Support all **six competition languages**.
* Have a vocabulary size of **10,000 tokens or fewer**.
* Be **built by you** from the provided competition training data. No pretrained tokenizers, no external corpora, no third-party APIs or services.
* Be submitted as a valid Hugging Face `tokenizer.json`.
* Keep English and French within **1.15 times** your average score on the four target languages.
* Finish evaluation within **5 times** the time the character-level baseline takes on the same machine.
* Pass the official submission checker.

You must also submit the notebook you used to build your tokenizer, as
`notebook.ipynb` in your team directory, by the end of the competition. **A team
that does not submit its notebook will be disqualified.**

### Build It Yourself

Your vocabulary must be learned from the competition training data by code you
wrote or configured. The point of the challenge is the design decisions, not
finding the best existing vocabulary.

**Allowed**

* The Hugging Face `tokenizers` library, including its trainers.
* Your own tokenization code, in any style, as long as the result loads as `tokenizer.json`.
* Any algorithm, normalizer, pre-tokenizer or sampling strategy you can express.

**Not allowed**

* Submitting or adapting a pretrained tokenizer, for example `AutoTokenizer.from_pretrained(...)`, or reusing a published vocabulary or merge table such as those from GPT-2, Llama, mBERT or NLLB.
* Any training text other than the provided competition data.
* Calling an external API, model or service at any point, whether while building your tokenizer or during evaluation.
* Downloading vocabularies, word lists, frequency tables or embeddings from anywhere else.

Evaluation runs offline with no network access, so a submission that depends on
an external resource cannot be scored. The notebook you submit at the end is how
we check that your tokenizer is your own work.

---

## Evaluation

Your tokenizer is evaluated on **how efficiently it represents text** in Hausa, Swahili, Yoruba, and Amharic.

### Token Fertility

For each language, we calculate **token fertility**:

$$F_l = \frac{\text{Number of tokens produced}}{\text{Number of words}}$$

For example, if 100 words are represented using 150 tokens:

$$F = \frac{150}{100} = 1.5$$

Lower fertility means the tokenizer needs fewer tokens to represent the same amount of text. A word is a whitespace-separated run of characters in the original text.

### Unknown Tokens

A tokenizer that cannot represent a word emits `[UNK]`. That is not efficiency, it is failure, so it is penalised heavily. For each language:

$$U_l = \frac{\text{Number of [UNK] tokens}}{\text{Number of words}}$$

$$S_l = F_l + 100\,U_l$$

The penalty is deliberately harsh:

| `[UNK]` rate | Added to that language's score |
| ---: | ---: |
| 0% | 0.00 |
| 0.1% | +0.10 |
| 1% | +1.00 |
| 5% | +5.00 |
| 10% | +10.00 |

Every tokenizer still receives a score. A word-level tokenizer can be submitted, but if it cannot represent unseen words it is hammered. Byte-level and subword approaches reach full coverage and compete on fertility alone.

### Final Score

$$\text{Score} = \frac{S_{\text{Hausa}} + S_{\text{Swahili}} + S_{\text{Yoruba}} + S_{\text{Amharic}}}{4}$$

**Lower is better.** Each language contributes equally.

So the objective is:

> **Represent the text with as few tokens as possible, while avoiding unknown tokens.**

### English and French

English and French are part of the training and evaluation data but are **not scored**. They act as a guardrail: neither may cost more than **1.15 times** your own average across the four target languages.

This exists because without it the winning move is to drop English and French entirely. Doing that scores about 4% better on the four target languages while making French roughly 50% more expensive, which is not a multilingual tokenizer.

### Baselines

We provide two simple baselines to help you get started:

| Baseline | Idea | Score | What it demonstrates |
| --- | --- | ---: | --- |
| **Word-level** | Each of the 10,000 most frequent words is one token | 34.0 | Fertility near 1.2, the best possible, but a 33% `[UNK]` rate destroys the score |
| **Character-level** | One token per character, byte fallback for the rest | 5.68 | Full coverage and no penalty, but many tokens per word |

These are the two extremes of tokenizer design, and both are in
[starter/baselines/](starter/baselines/). Word-level shows what the penalty is
for; character-level shows what full coverage costs. A good subword tokenizer
beats both.

Your challenge is to find something better than both. You might explore **BPE,
WordPiece, Unigram, byte-level approaches**, different pre-tokenization, or how
you sample the six languages while training.

---

## Making a Submission

1. **Build your tokenizer.** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aims-ai-research-foundations/airf-multilingual-tokenizer-challenge/blob/main/starter/starter.ipynb) or open [starter/starter.ipynb](starter/starter.ipynb) in Google Colab. It installs everything, loads the data, and trains two baselines you can improve on.
2. **Export it** as a single file named `tokenizer.json`.
3. **Check it** by running the submission checker at the end of the notebook. It applies the same rules as official evaluation and shows your fertility, `[UNK]` rate and score for every language.
4. **Fork this repository** and create a branch named exactly `submission`.

   ```bash
   git checkout -b submission
   ```

5. **Add one directory for your team**, named in lowercase kebab case, containing your `tokenizer.json` and a `metadata.yml` naming your team and members. An optional `README.md` can describe your approach.

   ```text
   submissions/<team-name>/
   ├── tokenizer.json
   └── metadata.yml
   ```

6. **Push the `submission` branch.** The automated check runs on every push to that branch and validates your entry.

   ```bash
   git add submissions/<team-name>
   git commit -m "<Your commit message>"
   git push origin submission
   ```

7. **Open a pull request** from your `submission` branch once the check passes. A scheduled workflow scores accepted entries on the hidden test set and updates the leaderboard.
8. **Keep improving.** Push as often as you like until the deadline. **Your most recent tokenizer is the one that gets judged**.
9. **Add your notebook before the deadline.** Commit the notebook you used to build your tokenizer as `notebook.ipynb` in the same directory. **Teams without a notebook are disqualified.**

Full pull request policy and the `metadata.yml` format are in [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Submission Checklist

- [ ] My tokenizer supports all six competition languages.
- [ ] My vocabulary contains no more than 10,000 tokens.
- [ ] My tokenizer was built by my own code from the provided training data only.
- [ ] I used no pretrained tokenizer, no external corpus, and no third-party API.
- [ ] I evaluated it on the validation set and reviewed the per-language results.
- [ ] My file is named `tokenizer.json` and loads with `tokenizers==0.22.1`.
- [ ] It needs no custom code or external resources during evaluation.
- [ ] I have checked my `[UNK]` rate, because each 1% adds 1.00 to my score.
- [ ] English and French stay within 1.15 times my four-language average.
- [ ] It passes the official submission checker.
- [ ] My team directory contains `tokenizer.json` and `metadata.yml`, and nothing else that is not allowed.
- [ ] My changes are on a branch named `submission` and the automated check passed.
- [ ] The tokenizer currently on my branch is the one I want judged, because the most recent one counts.
- [ ] I have committed `notebook.ipynb` to my team directory before the deadline.

---

## What Is in This Repository

```text
starter/
├── starter.ipynb          the notebook, self-contained, runs on Colab
├── utils.py               the submission checker
├── submission_checker.py  the same checks from the command line
└── baselines/             the word-level and character-level tokenizers
submissions/               one directory per team, added by pull request
competition/               the evaluation package used by CI
scripts/                   training, evaluation and leaderboard commands
tests/                     the contract tests, including small data fixtures
LEADERBOARD.md             regenerated nightly
leaderboard.json           the same standings as a feed, read by the website
```

The starter notebook needs nothing from this repository. It installs its own
dependencies, pulls the data from the Hugging Face Hub, and downloads the
checker. Cloning is only necessary if you want to run the evaluation code
yourself.

---

## Working Locally

Optional. [uv](https://docs.astral.sh/uv/) is the only supported package manager,
and Python 3.10 or newer is required. No GPU is needed.

```bash
uv sync --dev
uv run python starter/submission_checker.py submissions/<team>/tokenizer.json
uv run pytest
```

To rebuild the two reference baselines:

```bash
uv run python scripts/train_baselines.py --train-data <your train.csv>
```

---

## References

This challenge builds on concepts introduced in the **AI Research Foundations** learning path:

* [**Course 01: Build Your Own Small Language Model**](https://www.skills.google/course_templates/1341) introduces the language model pipeline and where tokenization fits into building a language model.
* [**Course 02: Represent Your Language Data**](https://www.skills.google/course_templates/1452) explores how text is represented for language models, including tokenization and vocabulary construction.

**AI Research Foundations Learning Path:**
https://www.skills.google/paths/3135

---

## License

Code in this repository is released under the [MIT License](LICENSE). The
competition text is derived from Wikipedia and remains under CC BY-SA 3.0 and
the GFDL; see the [dataset card](https://huggingface.co/datasets/Similoluwa/african-multilingual-tokenizer-challenge)
for attribution and licensing details.
