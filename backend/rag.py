import os
from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# load_dotenv()

# DB_PATH = os.getenv("CHROMA_DB_PATH", "chroma_db")
# COLLECTION_NAME = os.getenv("COLLECTION_NAME", "portfolio_data")


# def get_answer(question: str, history: str = "") -> str:
#     embeddings = MistralAIEmbeddings(
#         model="mistral-embed"
#     )

#     vectorstore = Chroma(
#         persist_directory=DB_PATH,
#         embedding_function=embeddings,
#         collection_name=COLLECTION_NAME
#     )

#     retriever = vectorstore.as_retriever(
#         search_kwargs={"k": 4}
#     )

#     docs = retriever.invoke(question)

#     context = "\n\n".join(doc.page_content for doc in docs)

#     llm = ChatMistralAI(
#         model="mistral-small-latest",
#         temperature=0.2
#     )

#     prompt = ChatPromptTemplate.from_messages([
#         (
#             "system",
#             """
# You are a portfolio chatbot for Nexora Technologies.

# Use only the provided company context to answer.

# If answer is not available in context, say:
# "I could not find this information in the company data."

# Keep answer short and helpful.
# """
#         ),
#         (
#             "human",
#             """
# Company Context:
# {context}

# Previous Chat History:
# {history}

# User Question:
# {question}
# """
#         )
#     ])

#     final_prompt = prompt.invoke({
#         "context": context,
#         "history": history,
#         "question": question
#     })

#     response = llm.invoke(final_prompt)

#     return response.content

import os
from dotenv import load_dotenv

from langchain_mistralai import (
    ChatMistralAI,
    MistralAIEmbeddings
)

from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

DB_PATH = os.getenv(
    "CHROMA_DB_PATH",
    "chroma_db"
)

COLLECTION_NAME = os.getenv(
    "COLLECTION_NAME",
    "portfolio_data"
)


def get_answer(
    question: str,
    history: str = ""
) -> str:

    # =========================
    # Embeddings
    # =========================

    embeddings = MistralAIEmbeddings(
        model="mistral-embed"
    )

    # =========================
    # Vector DB
    # =========================

    vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    # =========================
    # Retriever
    # =========================

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    # =========================
    # Retrieve docs
    # =========================

    docs = retriever.invoke(
        question,

        config={
            "run_name": "portfolio_retriever",

            "tags": [
                "retriever",
                "portfolio-chatbot",
                "mistral"
            ],

            "metadata": {
                "question": question
            }
        }
    )

    # =========================
    # Context
    # =========================

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    # =========================
    # LLM
    # =========================

    llm = ChatMistralAI(
        model="mistral-small-latest",
        temperature=0.2
    )

    # =========================
    # Prompt
    # =========================

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

    # =========================
    # Final Prompt
    # =========================

    final_prompt = prompt.invoke({
        "context": context,
        "history": history,
        "question": question
    })

    # =========================
    # LLM Response
    # =========================

    response = llm.invoke(

        final_prompt,

        config={

            "run_name": "portfolio_rag_answer",

            "tags": [
                "llm",
                "portfolio-chatbot",
                "rag",
                "mistral"
            ],

            "metadata": {
                "question": question
            }
        }
    )

    return response.content