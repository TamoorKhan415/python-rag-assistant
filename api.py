
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from app.services.answer_service import generate_answer
from app.services.retrieval_service import retrieve_relevant_chunks
from app.services.text_ingestion_service import ingest_text

app = FastAPI(
    title="Python RAG Assistant",
    version="1.0.0",
)




class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)

@app.post("/ask")
def ask_question(request: AskRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    # 1. Retrieve relevant chunks from ChromaDB
    chunks = retrieve_relevant_chunks(question)

    # 2. Send the question and retrieved chunks to Gemini
    answer = generate_answer(question, chunks)

    # 3. Return the generated answer and its sources
    return {
        "question": question,
        "answer": answer,
        "sources": chunks
    }


class AddTextRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=50000)


@app.post("/documents/text")
def add_text(request: AddTextRequest):
    title = request.title.strip()
    content = request.content.strip()

    if not title or not content:
        raise HTTPException(
            status_code=400,
            detail="Title and content cannot be empty.",
        )

    return ingest_text(title, content)
