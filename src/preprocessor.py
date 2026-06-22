import re
import string
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# Download NLTK stopwords on first use.
try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOPWORDS = set(stopwords.words("english"))


STEMMER = PorterStemmer()
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def lowercase_text(text):
    return text.lower()


def remove_punctuation(text):
    translator = str.maketrans(string.punctuation, " " * len(string.punctuation))
    return text.translate(translator)


def tokenize(text):
    return re.findall(r"\b[a-zA-Z0-9]+\b", text)


def remove_stopwords(tokens):
    return [token for token in tokens if token not in STOPWORDS]


def stem_tokens(tokens):
    return [STEMMER.stem(token) for token in tokens]


def preprocess_text(text, use_stemming=True):
    """
    Lowercase, tokenize, remove stopwords, and optionally stem.
    """
    text = lowercase_text(text)
    text = remove_punctuation(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)

    if use_stemming:
        tokens = stem_tokens(tokens)

    return tokens


def preprocess_file(file_path, use_stemming=True):
    file_path = Path(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    return preprocess_text(text, use_stemming=use_stemming)


def main():
    corpus_dir = PROJECT_ROOT / "data/corpus"
    sample_files = sorted(corpus_dir.glob("*.txt"))

    if not sample_files:
        print(f"No corpus files found in: {corpus_dir}")
        return

    sample_path = sample_files[0]

    tokens = preprocess_file(sample_path, use_stemming=True)

    print(f"Sample file: {sample_path}")
    print(f"Number of tokens after preprocessing: {len(tokens)}")
    print("\nFirst 50 tokens:")
    print(tokens[:50])


if __name__ == "__main__":
    main()