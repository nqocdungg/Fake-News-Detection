import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------------------------------------------------------------------
# Auto-download required NLTK resources (silent, only if missing)
# ---------------------------------------------------------------------------
for resource, path in [
    ("stopwords", "corpora/stopwords"),
    ("wordnet",   "corpora/wordnet"),
    ("omw-1.4",   "corpora/omw-1.4"),
    ("punkt",     "tokenizers/punkt"),
    ("punkt_tab", "tokenizers/punkt_tab"),
]:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource, quiet=True)

# ---------------------------------------------------------------------------
# Module-level singletons (initialised once on import)
# ---------------------------------------------------------------------------
_lemmatizer = WordNetLemmatizer()

_stop_words = set(stopwords.words("english"))
_negation_words = {"no", "not", "never", "neither", "nor"}
_stop_words -= _negation_words   # keep negation words so model understands "not true"

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess_text(text: str) -> str:
    """
    Full preprocessing pipeline for a single news article text.
    """

    if not isinstance(text, str):
        text = str(text)

    # 1. Remove Reuters source marker
    text = re.sub(r'\(Reuters\)|\bReuters\b', ' ', text, flags=re.IGNORECASE)

    # 2. Lowercase
    text = text.lower()

    # 3. Strip noise
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)   # URLs
    text = re.sub(r"\S+@\S+", " ", text)                   # e-mails
    text = re.sub(r"\d+", " ", text)                       # digits
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # 4-6. Tokenise → lemmatise → remove stopwords & short tokens
    tokens = [
        _lemmatizer.lemmatize(word)
        for word in text.split()
        if word not in _stop_words and len(word) > 2
    ]

    # 7. Re-join and strip extra whitespace
    return re.sub(r"\s+", " ", " ".join(tokens)).strip()