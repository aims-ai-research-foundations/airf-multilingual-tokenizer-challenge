# AI Research Foundations Multilingual Tokenization Challenge

**Six languages. One vocabulary. Make every token count.**

## Overview

Tokenization is the first step in how language models turn human language into something they can process, making the design of a tokenizer a fundamental part of building a language model.

But not every tokenizer represents every language equally efficiently.

In this challenge, you will build **one tokenizer for six languages** under a fixed vocabulary budget of **10,000 tokens**.

Your goal is to represent text efficiently across all six languages and beat the competition baseline.

The lower your score, the higher you climb on the leaderboard.

**Start here:** [starter/starter.ipynb](starter/starter.ipynb)

**Current standings:** [LEADERBOARD.md](LEADERBOARD.md), regenerated every night.

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

* **Training set:** Used to build your tokenizer.
* **Validation set:** Used to test your ideas and improve your tokenizer.
* **Test set:** Kept hidden and used to calculate the official leaderboard score.

Each example contains a language code and a piece of text.

| language | text                                           |
| -------- | ---------------------------------------------- |
| en       | Artificial intelligence is changing the world. |
| yo       | Ẹ̀kọ́ ṣe pàtàkì fún gbogbo ènìyàn.             |
| sw       | Teknolojia inabadilisha dunia.                 |

The train and validation splits are published on the Hugging Face Hub at
[`Similoluwa/african-multilingual-tokenizer-challenge`](https://huggingface.co/datasets/Similoluwa/african-multilingual-tokenizer-challenge).
The starter notebook loads them for you.

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
* Be trained using only the **provided competition training data**.
* Be submitted as a valid Hugging Face `tokenizer.json`.
* Work without custom code or external resources during evaluation.
* Pass the official submission checker.

---

## Evaluation

Your tokenizer is evaluated on how efficiently it represents text across all six languages.

### 1. Measure Token Fertility

For each language $l$, we calculate:

$$F_l = \frac{\text{Number of tokens produced}}{\text{Number of characters}}$$

Lower fertility means the tokenizer uses fewer tokens to represent the same amount of text.

### 2. Compare Against the Baseline

We normalize the fertility for each language against the official baseline tokenizer:

$$N_l = \frac{F_l}{F_l^{\text{baseline}}}$$

A normalized score below **1.0** means your tokenizer is more efficient than the baseline for that language.

### 3. Calculate the Final Score

Your competition score is the average normalized fertility across all six languages:

$$\text{Score} = \frac{1}{6} \sum_{l=1}^{6} N_l$$

Every language contributes **equally** to the final score.

|      Score | Meaning                  |
| ---------: | ------------------------ |
|   **1.00** | Same as the baseline     |
|   **0.90** | Approximately 10% better |
|   **0.80** | Approximately 20% better |
| **> 1.00** | Worse than the baseline  |

**Lower is better.**

Your validation score helps you develop your tokenizer. Your official leaderboard score is calculated separately on the hidden test set.

---

## Making a Submission

1. **Build your tokenizer.** Open [starter/starter.ipynb](starter/starter.ipynb) in Google Colab. It installs everything, loads the data, and trains two baselines you can improve on.
2. **Export it** as a single file named `tokenizer.json`.
3. **Check it** by running the submission checker at the end of the notebook. It applies the same validity rules as official evaluation.
4. **Fork this repository** and create a branch named exactly `submission`.

   ```bash
   git checkout -b submission
   ```

5. **Add one directory for your team**, named in lowercase kebab case, containing your `tokenizer.json` and a `metadata.yml` naming your team and members. An optional `README.md` can describe your approach.

   ```text
   submissions/team-tokenlab/
   ├── tokenizer.json
   └── metadata.yml
   ```

6. **Push the `submission` branch.** The automated check runs on every push to that branch and validates your entry.

   ```bash
   git add submissions/team-tokenlab
   git commit -m "Add TokenLab submission"
   git push origin submission
   ```

7. **Open a pull request** from your `submission` branch once the check passes. A scheduled workflow scores accepted entries on the hidden test set and updates the leaderboard.
8. **Mark your final entry.** You can keep pushing improvements to the same branch until the freeze time. When the deadline is announced, set `final: true` on the one tokenizer you want judged.

Full pull request policy and the `metadata.yml` format are in [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Submission Checklist

- [ ] My tokenizer supports all six competition languages.
- [ ] My vocabulary contains no more than 10,000 tokens.
- [ ] My tokenizer was trained using only the provided competition training data.
- [ ] I evaluated it on the validation set and reviewed the per-language results.
- [ ] My file is named `tokenizer.json` and loads with `tokenizers==0.22.1`.
- [ ] It needs no custom code or external resources during evaluation.
- [ ] It passes the official submission checker.
- [ ] My team directory contains `tokenizer.json` and `metadata.yml`, and nothing else that is not allowed.
- [ ] My changes are on a branch named `submission` and the automated check passed.
- [ ] I set `final: true` on the entry I want judged, once the deadline is announced.

---

## References

This challenge builds on concepts introduced in the **AI Research Foundations** learning path:

* [**Course 01: Build Your Own Small Language Model**](https://www.skills.google/course_templates/1341) introduces the language model pipeline and where tokenization fits into building a language model.
* [**Course 02: Represent Your Language Data**](https://www.skills.google/course_templates/1452) explores how text is represented for language models, including tokenization and vocabulary construction.

**AI Research Foundations Learning Path:**
https://www.skills.google/paths/3135
