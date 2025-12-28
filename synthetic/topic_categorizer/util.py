import chromadb
from pathlib import Path
from sentence_splitter import split  # noqa


MAX_DIFFERENCE = 1.3
MAX_DB_RESULTS = 10
with open('prompt.md', 'r', encoding='utf-8') as _f:
    PROMPT = _f.read()


def db_read(texts: list[str]):
    """
    Get results from ChromaDB based on vector similarity
    :param texts: a list of strings to search for
    :return: Query results directly from ChromaDB
    """
    client = chromadb.PersistentClient(path=Path(__file__).resolve().parent.parent.absolute().__str__() + 'database.chroma')
    collection = client.get_collection(name='topic_categorizer')
    return collection.query(query_texts=texts, n_results=MAX_DB_RESULTS)

