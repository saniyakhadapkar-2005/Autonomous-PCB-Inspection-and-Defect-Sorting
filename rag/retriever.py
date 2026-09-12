from pathlib import Path

from langchain_chroma import Chroma

from rag.embeddings import get_embeddings


VECTOR_DB_PATH = "data/chroma_db"


def get_vector_store():
    """
    Load the existing ChromaDB vector store.
    """

    if not Path(VECTOR_DB_PATH).exists():
        raise FileNotFoundError(
            f"Vector database not found: {VECTOR_DB_PATH}\n"
            "Run: python -m rag.vector_store"
        )

    embeddings = get_embeddings()

    vector_store = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings
    )

    return vector_store


def retrieve_relevant_knowledge(query, k=3):
    """
    Retrieve the most relevant PCB repair knowledge.
    """

    vector_store = get_vector_store()

    documents = vector_store.similarity_search(
        query,
        k=k
    )

    results = []

    for document in documents:
        results.append({
            "content": document.page_content,
            "metadata": document.metadata
        })

    return results


if __name__ == "__main__":

    print("=" * 70)
    print("PCB REPAIR KNOWLEDGE RETRIEVAL TEST")
    print("=" * 70)

    query = "How should a solder bridge on a PCB be repaired?"

    print(f"\nQuery:")
    print(query)

    print("\nSearching knowledge base...")

    results = retrieve_relevant_knowledge(query, k=3)

    print(f"\n✓ Retrieved {len(results)} relevant chunks\n")

    for index, result in enumerate(results, start=1):

        print("-" * 70)
        print(f"RESULT {index}")
        print("-" * 70)

        print(result["content"])

    print("\n" + "=" * 70)
    print("✓ RETRIEVAL TEST COMPLETED")
    print("=" * 70)