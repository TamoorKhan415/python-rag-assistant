
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
    collection_name="developer_knowledge",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)


def retrieve_relevant_chunks(question: str, k: int = 2):
    results = vector_store.similarity_search(
        question,
        k=k,
    )

    return [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
        }
        for doc in results
    ]
