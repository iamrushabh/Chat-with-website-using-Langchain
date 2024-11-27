import json
import os
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint

# Paths
TEXT_FOLDER = "botpenguin_folder"
EMBEDDINGS_FOLDER = "embeddings"
FAISS_INDEX_PATH = os.path.join(EMBEDDINGS_FOLDER, "embeddings.faiss")


# Step 1: Load FAISS index
def load_faiss_index(faiss_index_path, embeddings_model):
    if os.path.exists(faiss_index_path):
        print("Loading FAISS index from disk...")
        return FAISS.load_local(faiss_index_path, embeddings_model, allow_dangerous_deserialization=True)
    else:
        raise FileNotFoundError(f"FAISS index file not found at {faiss_index_path}")


# Step 2: Set up retriever
def setup_retriever(faiss_index):
    return faiss_index.as_retriever(search_type="similarity", search_kwargs={"k": 6})


# Step 3: Configure HuggingFace LLM
def initialize_llm():
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = "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"
    return HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        max_length=3000,
        temperature=0.2
    )


# Step 4: Define RAG chain
def setup_rag_chain(retriever, llm):
    template = """You are an AI assistant designed to answer questions based on the content of a website. You have access to context from this website, which contains important information. Your task is to extract relevant information from this context to answer the question at the end.

Always be concise, accurate, and relevant to the information found on the website. If you don't find enough information in the context to answer the question, say "I don't know" and avoid making up an answer.

Here is the context from the website:

{context}

Question: {question}

Provide a helpful, concise, and accurate answer based on the website content.
Dont give answer on your own just give the answer what is present in the website.
Thanks for asking!"""
    custom_rag_prompt = PromptTemplate.from_template(template)

    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )


# Main Execution
if __name__ == "__main__":
    # Initialize HuggingFace Embeddings model
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load the FAISS index
    faiss_index = load_faiss_index(FAISS_INDEX_PATH, embeddings_model)

    # Set up retriever
    retriever = setup_retriever(faiss_index)

    # Configure LLM
    llm = initialize_llm()

    # Create RAG chain
    rag_chain = setup_rag_chain(retriever, llm)

    # Ask a question
    question = "Which solutions are provided by botpenguin? give details of custom development services."
    response = rag_chain.invoke(question)
    print(response)
