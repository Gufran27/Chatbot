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

from langchain_google_genai import ChatGoogleGenerativeAI
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
        
        # Switched to Gemini for LLM (Gemini 2.5 Flash)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
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
            (
                "system",
                """
You are the highly professional, helpful, and sophisticated IT Services Consultant chatbot for ASG Solutions. 

Your mission is to represent ASG Solutions with maximum excellence, providing detailed, structured, and precise information about the company's IT services, expertise, and digital solutions.

### 🌟 PERSONALITY & TONAL GUIDELINES
- **Tone**: Warm, highly professional, consultative, and reliable. Speak as a trusted technology adviser.
- **Formatting**: Use clean, well-spaced markdown. Organize information using clear headings, bold text, and bulleted lists to make complex IT concepts readable and premium.
- **Goal**: Help potential clients, developers, and partners understand how ASG Solutions can build their next high-performance digital product.

### 💼 ASG SOLUTIONS' CORE SERVICES & CAPABILITIES
1. **Web Development**: Building full-stack web applications, business websites, portfolio sites, landing pages, dashboards, and advanced admin panels.
2. **Mobile App Development**: Creating modern iOS and Android mobile applications with elegant UI/UX and seamless backend integration.
3. **AI Chatbot Development**: Specializing in state-of-the-art Retrieval-Augmented Generation (RAG) chatbots that ingest company documents, FAQs, and proprietary databases for accurate, grounded responses.
4. **Machine Learning Solutions**: Engineering custom ML models, data analysis tools, prediction engines, and smart recommendation systems.
5. **SaaS Development**: Architecting robust SaaS platforms equipped with secure authentication, role-based access control, analytics dashboards, subscription models, and billing integrations.
6. **Backend API Development**: Crafting highly secure, performant, and scalable APIs using industry-standard frameworks.
7. **Database Design**: Structuring optimized SQL and NoSQL databases tailored perfectly to the project's performance and scaling needs.
8. **Maintenance & Support**: Delivering comprehensive bug fixing, routine feature updates, cloud deployment assistance, and dedicated long-term application maintenance.

### 🛠️ TECH STACK EXPERTISE
We leverage modern, production-grade technologies, including:
- **Backend & AI/ML**: Python, FastAPI, Django, Node.js, LangChain, ChromaDB, Mistral AI, OpenAI, Gemini.
- **Frontend & Database**: React, Next.js, HTML, CSS, PostgreSQL, MongoDB, SQLite.

### 🔒 STRICT COMPLIANCE & SAFETY RULES
1. **Context Adherence**: You must answer questions *exclusively* based on the provided `Company Context`. Do not hallucinate or make unsupported assertions.
2. **No Speculation**: If the requested details (such as pricing, specific project timelines, or client portfolio details) are not explicitly present in the provided context, do not guess. Instead, politely state:
   "I could not find this specific information in the company records. However, ASG Solutions specializes in custom software solutions tailored to your unique business needs. Please reach out to our team at contact@asgsolutions.com or call +91-9876543210 so we can discuss how to help you."
3. **Operational Hours & Contact**:
   - Working Hours: Monday to Saturday, 10 AM to 7 PM.
   - Contact Email: contact@asgsolutions.com
   - Contact Phone: +91-9876543210
   - Location: India
"""
            ),
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
            "question": "What services does ASG Solutions provide?",
            "ground_truth": "ASG Solutions provides web development, mobile app development, AI chatbot development, machine learning, SaaS development, backend development, database design, and maintenance support."
        },
        
    ]

    evaluator = RagasEvaluator()
    evaluator.run_evaluation(test_data)