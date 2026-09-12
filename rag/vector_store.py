from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.embeddings import get_embeddings


KNOWLEDGE_FILE = "knowledge_base/pcb_repair_knowledge.txt"
VECTOR_DB_PATH = "data/chroma_db"


def create_vector_store():

    print("=" * 70)
    print("CREATING PCB KNOWLEDGE VECTOR DATABASE")
    print("=" * 70)

    if not Path(KNOWLEDGE_FILE).exists():
        raise FileNotFoundError(
            f"Knowledge file not found: {KNOWLEDGE_FILE}"
        )

    print("\nLoading knowledge base...")

    loader = TextLoader(
        KNOWLEDGE_FILE,
        encoding="utf-8"
    )

    documents = loader.load()

    print(f"✓ Loaded {len(documents)} document(s)")

    print("\nSplitting documents...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    print(f"✓ Created {len(chunks)} chunks")

    print("\nCreating embeddings...")

    embeddings = get_embeddings()

    print("✓ Embedding model ready")

    print("\nCreating ChromaDB...")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH
    )

    print("✓ ChromaDB created")

    print("\n" + "=" * 70)
    print("✓ VECTOR DATABASE READY")
    print("=" * 70)

    return vector_store


if __name__ == "__main__":
    create_vector_store()