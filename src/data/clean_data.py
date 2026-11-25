
import re
import pandas as pd


def clean_sentence(sentence, min_words=8):
    if not isinstance(sentence, str):
        return None

    sentence = re.sub(r"\s+", " ", sentence).strip()

    if not sentence:
        return None

    if not re.search(r"[^\W\d_]", sentence):
        return None

    if len(sentence.split()) < min_words:
        return None

    return sentence


def clean_corpus(texts, min_words=8):
    cleaned = []
    seen = set()

    for sentence in texts:
        sentence = clean_sentence(sentence, min_words)

        if sentence is None:
            continue

        normalized = sentence.casefold()

        if normalized in seen:
            continue

        seen.add(normalized)
        cleaned.append(sentence)

    return cleaned
