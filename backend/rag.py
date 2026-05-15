import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

DB_PATH = os.getenv("CHROMA_DB_PATH", "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "portfolio_data")

def get_answer(question: str, history: str = "") -> str:
    # Switched to HuggingFace (Local) embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Vector DB
    vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    # Retriever
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    # Retrieve docs
    docs = retriever.invoke(
        question,
        config={
            "run_name": "portfolio_retriever",
            "tags": ["retriever", "portfolio-chatbot", "groq"],
            "metadata": {"question": question}
        }
    )

    # Context
    context = "\n\n".join([doc.page_content for doc in docs])

    # LLM (Groq llama-3.1-8b-instant)
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2
    )

    # Prompt
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are Nexora Technologies portfolio chatbot.

Rules:
- Answer only from the provided company context.
- Keep answers short and professional.
- Do not make fake claims.
- If answer is not found in context say:
  "I could not find this information in the company data."
"""
        ),
        (
            "human",
            """
Company Context:
{context}

Previous Chat History:
{history}

User Question:
{question}
"""
        )
    ])

    # Final Prompt
    final_prompt = prompt.invoke({
        "context": context,
        "history": history,
        "question": question
    })

    # LLM Response
    response = llm.invoke(
        final_prompt,
        config={
            "run_name": "portfolio_rag_answer",
            "tags": ["llm", "portfolio-chatbot", "rag", "groq"],
            "metadata": {"question": question}
        }
    )

    return response.content