from langchain_ollama import OllamaEmbeddings


def get_embeddings():
    """
    Create local embedding model using Ollama.
    """

    return OllamaEmbeddings(
        model="nomic-embed-text"
    )