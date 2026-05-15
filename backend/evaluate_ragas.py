import os
from dotenv import load_dotenv
from datasets import Dataset
import pandas as pd
from typing import List, Dict

# Ragas imports (v0.2+ style)
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Configuration
DB_PATH = os.getenv("CHROMA_DB_PATH", "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "portfolio_data")

class RagasEvaluator:
    def __init__(self):
        # Switched to HuggingFace (Local) for embeddings to avoid Mistral API errors
        print("Loading HuggingFace embeddings (local)...")
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Switched to Groq for LLM (Llama 3.1 8B)
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0
        )
        
        self.vectorstore = Chroma(
            persist_directory=DB_PATH,
            embedding_function=self.embeddings,
            collection_name=COLLECTION_NAME
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})

    def get_rag_response(self, question: str) -> Dict:
        """Runs the RAG pipeline for a single question."""
        docs = self.retriever.invoke(question)
        contexts = [doc.page_content for doc in docs]
        context_text = "\n\n".join(contexts)

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are Nexora Technologies portfolio chatbot. Use only the provided company context to answer. If answer is not present, say: 'I could not find this information in the company data.'"),
            ("human", "Company Context:\n{context}\n\nQuestion:\n{question}")
        ])

        final_prompt = prompt.invoke({"context": context_text, "question": question})
        response = self.llm.invoke(final_prompt)

        return {
            "answer": response.content,
            "contexts": contexts
        }

    def run_evaluation(self, test_data: List[Dict]):
        """Runs evaluation on the provided test data."""
        print(f"Starting evaluation on {len(test_data)} samples using GROQ (llama-3.1-8b-instant)...")
        
        questions = []
        answers = []
        contexts = []
        ground_truths = []

        for i, item in enumerate(test_data):
            print(f"[{i+1}/{len(test_data)}] Processing: {item['question']}")
            try:
                res = self.get_rag_response(item["question"])
                
                questions.append(item["question"])
                answers.append(res["answer"])
                contexts.append(res["contexts"])
                ground_truths.append(item["ground_truth"])
            except Exception as e:
                print(f"Error processing question: {e}")

        if not questions:
            print("No questions were processed successfully. Check if your Vector DB exists and is using the same embeddings.")
            return

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        })

        print("Evaluating with Ragas metrics (using GROQ as evaluator)...")
        result = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            llm=self.llm,
            embeddings=self.embeddings
        )

        print("\nEvaluation Results:")
        print(result)

        # Save results
        df = result.to_pandas()
        output_file = "ragas_report.csv"
        df.to_csv(output_file, index=False)
        print(f"\nDetailed report saved to {output_file}")
        
        return result

if __name__ == "__main__":
    # Sample test data
    test_data = [
        {
            "question": "What services does Nexora Technologies provide?",
            "ground_truth": "Nexora Technologies provides web development, mobile app development, AI chatbot development, machine learning, SaaS development, backend development, database design, and maintenance support."
        },
        
    ]

    evaluator = RagasEvaluator()
    evaluator.run_evaluation(test_data)