import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data")

def generate_rag_testset():
    print("Loading documents from:", DATA_PATH)
    loader = DirectoryLoader(
        DATA_PATH,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()

    # Initialize LLM (Groq) and Embeddings (HuggingFace Local)
    generator_llm = ChatGroq(model="llama-3.1-8b-instant") 
    generator_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Initialize Generator
    generator = TestsetGenerator.from_langchain(
        generator_llm,
        generator_llm, 
        generator_embeddings
    )

    # Generate testset
    print("Generating testset using Groq and Local Embeddings...")
    testset = generator.generate_with_langchain_docs(
        documents,
        test_size=10,
        distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25}
    )

    # Save to CSV
    output_file = "generated_testset.csv"
    testset.to_pandas().to_csv(output_file, index=False)
    print(f"Testset generated and saved to {output_file}")

if __name__ == "__main__":
    generate_rag_testset()
