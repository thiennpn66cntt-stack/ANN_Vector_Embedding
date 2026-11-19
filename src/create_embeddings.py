
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data/processed")
EMBED_DIR = Path("embeddings")

MODEL_NAME = "intfloat/multilingual-e5-small"


def read_sentences(path):

    with open(path, "r", encoding="utf-8") as file:

        return [
            line.strip()
            for line in file
            if line.strip()
        ]


def main():

    EMBED_DIR.mkdir(parents=True, exist_ok=True)

    print("Dang tai mo hinh...")

    model = SentenceTransformer(MODEL_NAME)

    database = read_sentences(
        DATA_DIR / "database.txt"
    )

    queries = read_sentences(
        DATA_DIR / "queries.txt"
    )

    database_input = [
        "passage: " + text
        for text in database
    ]

    query_input = [
        "query: " + text
        for text in queries
    ]

    print("Dang tao embedding database...")

    db_embeddings = model.encode(
        database_input,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    print("Dang tao embedding query...")

    query_embeddings = model.encode(
        query_input,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    db_embeddings = db_embeddings.astype(np.float32)
    query_embeddings = query_embeddings.astype(np.float32)

    np.save(
        EMBED_DIR / "database_embeddings.npy",
        db_embeddings
    )

    np.save(
        EMBED_DIR / "query_embeddings.npy",
        query_embeddings
    )

    print("Hoan thanh tao embedding!")

    print("Database:", db_embeddings.shape)
    print("Query:", query_embeddings.shape)


if __name__ == "__main__":
    main()
