# from langchain.chains import RetrievalQA
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.llms import HuggingFaceEndpoint
# from langchain.prompts import PromptTemplate
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.document_loaders import TextLoader
# from langchain.vectorstores import Chroma
# from langchain.vectorstores import DocArrayInMemorySearch
# import os

# # Set Hugging Face API Token
# os.environ['HUGGINGFACEHUB_API_TOKEN'] = "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"

# # Initialize the LLM
# llm = HuggingFaceEndpoint(
#     endpoint_url="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
#     headers={"Authorization": f"Bearer {os.getenv('HUGGINGFACEHUB_API_TOKEN')}"},
#     max_length=3000,
#     temperature=0.3,
# )

# # Function to load and process text data
# def load_and_process_text(file_path, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
#     """Load text data, split, and embed with DocArray."""
#     # Load text data
#     loader = TextLoader(file_path, encoding="utf-8")
#     documents = loader.load()

#     # Split text into chunks
#     text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     texts = text_splitter.split_documents(documents)

#     # Embed texts using DocArrayInMemorySearch
#     embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
#     docsearch = DocArrayInMemorySearch.from_documents(texts, embeddings)

#     print("Document search index created.")
#     return docsearch

# # Create RetrievalQA Chain
# def create_retrieval_chain(docsearch, llm):
#     """Create a RetrievalQA chain using DocArray and LLM."""
#     retriever = docsearch.as_retriever()
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=retriever,
#         return_source_documents=True,
#         chain_type="stuff",
#     )
#     return qa_chain

# # Process user input
# def process_user_query(qa_chain, query):
#     """Get an answer to a user's query."""
#     response = qa_chain.invoke({"query": query})
#     print("\nAnswer:")
#     print(response["result"])

# # Main Workflow
# def main():
#     # Path to text data
#     data_file = r"botpenguin_content.txt"

#     # Load and process text data
#     docsearch = load_and_process_text(data_file)

#     # Create QA chain
#     qa_chain = create_retrieval_chain(docsearch, llm)

#     print("Chatbot ready! Ask me anything about the website.")
#     while True:
#         query = input("\nYou: ")
#         if query.lower() in {"exit", "quit"}:
#             print("Goodbye!")
#             break

#         # Get answer
#         process_user_query(qa_chain, query)

# if __name__ == "__main__":
#     main()



from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.vectorstores import DocArrayInMemorySearch
import os

# Set Hugging Face API Token
api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not api_token:
    api_token = input("Enter your Hugging Face API Token: ")
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_token

# Initialize the LLM
llm = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
    headers={"Authorization": f"Bearer {api_token}"},
    max_length=3000,
    temperature=0.3,
)

# Function to load and process text data
def load_and_process_text(file_path, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
    """Load text data, split recursively, and embed with DocArray."""
    try:
        # Load text data
        loader = TextLoader(file_path, encoding="utf-8")
        documents = loader.load()

        # Split text using RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Maximum tokens per chunk
            chunk_overlap=50,  # Overlap between chunks
            separators=["\n\n", "\n", ".", " ", ""],  # Hierarchy of delimiters
        )
        texts = text_splitter.split_documents(documents)

        if not texts:
            raise ValueError("No valid text found after splitting. Check the input document.")

        # Embed texts using DocArrayInMemorySearch
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        docsearch = DocArrayInMemorySearch.from_documents(texts, embeddings)

        print("Document search index created successfully.")
        return docsearch
    except Exception as e:
        print(f"Error processing text data: {e}")
        return None

# Create RetrievalQA Chain
def create_retrieval_chain(docsearch, llm):
    """Create a RetrievalQA chain using DocArray and LLM."""
    try:
        retriever = docsearch.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            chain_type="stuff",
        )
        return qa_chain
    except Exception as e:
        print(f"Error creating RetrievalQA chain: {e}")
        return None

# Process user input
def process_user_query(qa_chain, query):
    """Get an answer to a user's query."""
    try:
        response = qa_chain.invoke({"query": query})
        print("\nAnswer:")
        print(response["result"])
    except Exception as e:
        print(f"Error processing query: {e}")

# Main Workflow
def main():
    # Path to text data
    data_file = r"botpenguin_content.txt"

    # Load and process text data
    docsearch = load_and_process_text(data_file)
    if not docsearch:
        print("Failed to load and process text data. Exiting.")
        return

    # Create QA chain
    qa_chain = create_retrieval_chain(docsearch, llm)
    if not qa_chain:
        print("Failed to create QA chain. Exiting.")
        return

    print("Chatbot ready! Ask me anything about the website.")
    try:
        while True:
            query = input("\nYou: ")
            if query.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break

            # Get answer
            process_user_query(qa_chain, query)
    except KeyboardInterrupt:
        print("\nExiting... Goodbye!")

if __name__ == "__main__":
    main()
