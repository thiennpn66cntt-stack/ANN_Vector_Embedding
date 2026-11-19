
import re
import random
from pathlib import Path

RAW_FILE = Path("data/raw/sentences.txt")
OUTPUT_DIR = Path("data/processed")

TOTAL_SENTENCES = 50000
NUM_QUERIES = 500
SEED = 42


def clean_sentence(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            "Khong tim thay data/raw/sentences.txt"
        )

    sentences = []

    with open(RAW_FILE, "r", encoding="utf-8") as file:

        for line in file:

            sentence = clean_sentence(line)

            if len(sentence) >= 10:
                sentences.append(sentence)

    # Loai bo cau trung lap
    sentences = list(dict.fromkeys(sentences))

    if len(sentences) < TOTAL_SENTENCES:
        raise ValueError(
            f"Chi co {len(sentences)} cau. "
            f"Can it nhat {TOTAL_SENTENCES} cau."
        )

    random.seed(SEED)
    random.shuffle(sentences)

    sentences = sentences[:TOTAL_SENTENCES]

    database = sentences[:-NUM_QUERIES]
    queries = sentences[-NUM_QUERIES:]

    with open(
        OUTPUT_DIR / "database.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write("\n".join(database))

    with open(
        OUTPUT_DIR / "queries.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write("\n".join(queries))

    print("Da chia du lieu thanh cong!")
    print("Database:", len(database))
    print("Queries:", len(queries))


if __name__ == "__main__":
    main()
