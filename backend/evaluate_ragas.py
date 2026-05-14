import os
from dotenv import load_dotenv
import pandas as pd

from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

DB_PATH = os.getenv("CHROMA_DB_PATH", "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "portfolio_data")


def run_rag(question: str):
    embeddings = MistralAIEmbeddings(model="mistral-embed")

    vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    docs = retriever.invoke(question)

    contexts = [doc.page_content for doc in docs]
    context_text = "\n\n".join(contexts)

    llm = ChatMistralAI(
        model="mistral-small-latest",
        temperature=0.2
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are Nexora Technologies portfolio chatbot.

Use only the provided company context to answer.
If answer is not present, say:
"I could not find this information in the company data."
"""
        ),
        (
            "human",
            """
Company Context:
{context}

Question:
{question}
"""
        )
    ])

    final_prompt = prompt.invoke({
        "context": context_text,
        "question": question
    })

    response = llm.invoke(final_prompt)

    return response.content, contexts


test_data = [
    {
        "question": "What services does Nexora Technologies provide?",
        "ground_truth": "Nexora Technologies provides web development, mobile app development, AI chatbot development, machine learning, SaaS development, backend development, database design, and maintenance support."
    },
    {
        "question": "What are the working hours of Nexora Technologies?",
        "ground_truth": "Nexora Technologies works Monday to Saturday, 10 AM to 7 PM."
    },
    {
        "question": "Does Nexora Technologies build AI chatbots?",
        "ground_truth": "Yes, Nexora Technologies builds RAG-based AI chatbots that answer from company data, documents, FAQs, and knowledge bases."
    },
    {
        "question": "What is the contact email of Nexora Technologies?",
        "ground_truth": "The contact email is contact@nexora.com."
    }
]


def main():
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for item in test_data:
        question = item["question"]
        ground_truth = item["ground_truth"]

        answer, retrieved_contexts = run_rag(question)

        questions.append(question)
        answers.append(answer)
        contexts.append(retrieved_contexts)
        ground_truths.append(ground_truth)

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    evaluator_llm = ChatMistralAI(
        model="mistral-small-latest",
        temperature=0
    )

    evaluator_embeddings = MistralAIEmbeddings(
        model="mistral-embed"
    )

    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )

    print(result)

    df = result.to_pandas()
    df.to_csv("ragas_report.csv", index=False)

    print("Ragas report saved: ragas_report.csv")


if __name__ == "__main__":
    main()