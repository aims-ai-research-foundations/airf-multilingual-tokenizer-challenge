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
MAX_VOCAB_SIZE = 10_000
MAX_TOKENIZER_BYTES = 20 * 1024 * 1024
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

