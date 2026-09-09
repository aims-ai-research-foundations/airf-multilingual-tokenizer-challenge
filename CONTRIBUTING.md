# How to submit

You submit one file, `tokenizer.json`, through a pull request. There are no
predictions to upload and no code of yours is executed during evaluation.

## 1. Build your tokenizer

Open [starter/starter.ipynb](starter/starter.ipynb) in Google Colab. It installs
everything it needs, loads the competition data, trains two starter tokenizers,
and scores them on the validation split.

## 2. Check it before you open a pull request

At the end of the notebook, run the submission checker:

```python
from utils import profile_submission

profile_submission("tokenizer.json", data=validation)
```

From a clone, the same checks run from the command line:

```bash
uv run python starter/submission_checker.py path/to/tokenizer.json
```

Your tokenizer must:

- be a single file named `tokenizer.json`, at most 20 MiB;
- load with Hugging Face `tokenizers==0.22.1`;
- have at most 10,000 entries by `get_vocab_size(with_added_tokens=True)`;
- produce at least one token and a non-empty decode in all six languages;
- need no external files, network access, or custom code;
- be built by your own code from the provided training data, with no
  pretrained tokenizer, external corpus, or third-party API involved.

## 3. Add your submission

Fork this repository and create a branch named exactly `submission`. The
automated check runs on every push to that branch, and nothing else triggers it:

```bash
git checkout -b submission
```

Then create one directory for your team, named in lowercase kebab case:

```text
submissions/
└── <team-name>/
    ├── tokenizer.json     required
    ├── metadata.yml       required
    ├── notebook.ipynb     required before the deadline
    └── README.md          optional, describe your approach
```

`metadata.yml` looks like this:

```yaml
team: <Your Team Name>
members:
  - <Member One>
  - <Member Two>
affiliation: <Optional organization>
approach: <Short public description of your tokenizer>
```

One team owns exactly one directory. Nothing else may be added to it: no
archives, no symlinks, no model sidecars.

## 4. Push the branch, then open a pull request

```bash
git add submissions/<team-name>
git commit -m "<Your commit message>"
git push origin submission
```

Pushing `submission` runs the validation workflow. It checks that your branch
changes exactly one team directory, that the directory contains only permitted
files, and that the tokenizer itself passes the validity contract. Everything
your branch adds relative to `main` is treated as the submission, so keep
unrelated changes off it.

Once the check passes, open a pull request from your `submission` branch. A
scheduled workflow then scores every accepted submission on the hidden test
split and updates the leaderboard.

You may keep pushing improvements to the same branch until the deadline. Your
most recent tokenizer is the one that gets judged, so there is nothing to mark
as final.

Before the deadline, commit the notebook you used to build your tokenizer as
`notebook.ipynb` in your team directory. A team that does not submit its
notebook will be disqualified.

If two tokenizers reach the same score, throughput breaks the tie and the
fastest one wins.

## Contributing to the tooling

For changes to the evaluation package, scripts, or notebook rather than a
competition entry, open a separate issue and pull request that contains no
submission.
