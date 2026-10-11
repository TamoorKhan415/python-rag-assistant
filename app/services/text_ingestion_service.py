
from uuid import uuid4

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.services.retrieval_service import vector_store


def ingest_text(title: str, content: str) -> dict:
    # 1. Prepare the document
    document = Document(
        page_content=content,
        metadata={
            "source": title,
            "type": "text",
        },
    )

    # 2. Split text into smaller chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    chunks = splitter.split_documents([document])

    if not chunks:
        raise ValueError("No text available to ingest.")

    # 3. Generate unique IDs for each chunk
    ids = [str(uuid4()) for _ in chunks]

    # 4. Generate embeddings and store in ChromaDB
    vector_store.add_documents(
        documents=chunks,
        ids=ids,
    )

    return {
        "title": title,
        "chunks_stored": len(chunks),
        "message": "Text successfully stored in ChromaDB.",
    }
