import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
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

    # LLM (Google Gemini 2.5 Flash)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2
    )

    # Prompt
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
            "tags": ["llm", "portfolio-chatbot", "rag", "google-genai"],
            "metadata": {"question": question}
        }
    )

    return response.content