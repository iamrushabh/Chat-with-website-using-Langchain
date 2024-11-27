import json
import os
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint
#from langchain.schema import format_documents as format_docs
#from langchain.llms.huggingface import HuggingFaceEndpoint
os.environ['HUGGINGFACEHUB_API_TOKEN'] = "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"
# Step 2: Load data from a single text file
def load_single_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()
def main():
    text_file_path = "/content/drive/MyDrive/botpenguin_content.txt"
    text_data = load_single_text_file(text_file_path)

    # Step 3: Split the text data into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )

    chunked_texts = text_splitter.split_text(text_data)

    # Step 5: Load Sentence Transformer embeddings model
    embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-large")

    # Step 6: Create FAISS vector store
    docsearch = FAISS.from_texts(chunked_texts, embeddings)

    # Step 7: Set up retriever
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    def format_docs(docsearch):
        return "\n\n".join(doc.page_content for doc in docsearch)

    # Step 8: Configure HuggingFace LLM

    llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2", max_length=3000, temperature=0.3)

    # Step 9: Define custom prompt template
    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    Always say "thanks for asking!" at the end of the answer.

    {context}

    Question: {question}

    Helpful Answer:"""
    custom_rag_prompt = PromptTemplate.from_template(template)

    # Step 10: Create RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )

    # Step 11: Invoke the RAG chain with a question
    response = rag_chain.invoke("What is Task Decomposition?")
    print(response)

if __name__ == "__main__":
    main()