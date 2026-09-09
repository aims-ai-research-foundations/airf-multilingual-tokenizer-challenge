from pathlib import Path

LANGUAGES = ("en", "fr", "ha", "sw", "yo", "am")
LANGUAGE_NAMES = {
    "en": "English",
    "fr": "French",
    "ha": "Hausa",
    "sw": "Swahili",
    "yo": "Yoruba",
    "am": "Amharic",
}
# Only these four languages contribute to the score.
SCORED_LANGUAGES = ("ha", "sw", "yo", "am")
# English and French are trained and evaluated but not scored. They act as a
# guardrail: neither may cost more than this multiple of a submission's own
# average across the scored languages. Without it, the winning strategy is to
# discard English and French entirely, which defeats the multilingual premise.
CONTEXT_LANGUAGES = ("en", "fr")
CONTEXT_FERTILITY_RATIO = 1.15

# Unknown tokens are penalised rather than disqualifying. One percent of
# words falling back to [UNK] adds 1.00 to that language's score.
UNKNOWN_PENALTY = 100.0

MAX_VOCAB_SIZE = 10_000
MAX_TOKENIZER_BYTES = 20 * 1024 * 1024
# A submission may take at most this multiple of the reference evaluation time.
MAX_TIME_MULTIPLE = 5.0
SUPPORTED_TOKENIZERS_VERSION = "0.22.1"
ROOT = Path(__file__).resolve().parents[1]
SMOKE_TEXTS = {
    "en": "Knowledge grows when it is shared.",
    "fr": "Le savoir grandit lorsqu’il est partagé.",
    "ha": "Ilimi yana ƙaruwa idan an raba shi.",
    "sw": "Maarifa hukua yanaposhirikishwa.",
    "yo": "Ìmọ̀ ń pọ̀ sí i nígbà tí a bá pín in.",
    "am": "እውቀት ሲካፈል ያድጋል።",
}
