# ------------------------- IMPORT LIBRARIES -------------------------

from dotenv import load_dotenv
import os
import pickle

import streamlit as st

# LangChain Components
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
    HuggingFaceEmbeddings,
)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS


# ------------------------- LOAD ENVIRONMENT VARIABLES -------------------------

# Load environment variables from .env (for local development)
load_dotenv()

# Get Hugging Face API token
# Streamlit Secrets (Cloud deployment)
HF_TOKEN = st.secrets.get("HF_TOKEN")

# Stop execution if token is missing
if not HF_TOKEN:
    st.error("Hugging Face API token not found.")
    st.stop()

# ------------------------- INITIALIZE LLM -------------------------

# Load Hugging Face Llama-3 model
llm_endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    temperature=0.9
)

# Wrap endpoint with LangChain Chat interface
llm = ChatHuggingFace(llm=llm_endpoint)


# ------------------------- STREAMLIT UI -------------------------

st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

# Take maximum 3 news URLs from user
urls = []

for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

# File used to save FAISS vector database
file_path = "faiss_store_hfai.pkl"

# Button to process the URLs
process_url_clicked = st.sidebar.button("Process URLs")

# Placeholder used to display progress messages
main_placeholder = st.empty()


# ------------------------- PROCESS NEWS ARTICLES -------------------------

if process_url_clicked:

    # Load article content from URLs
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    main_placeholder.text("Loading Articles... ✅")

    # Split large articles into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ","],
        chunk_size=1000
    )

    docs = text_splitter.split_documents(data)

    main_placeholder.text("Splitting Documents... ✅")

    # Generate embeddings for every text chunk
    embeddings = HuggingFaceEmbeddings()

    # Create FAISS Vector Store
    vectorstore = FAISS.from_documents(docs, embeddings)

    main_placeholder.text("Building Vector Database... ✅")

    # Save vector store locally to avoid recomputing embeddings
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)

    main_placeholder.text("Processing Completed ✅")


# ------------------------- QUESTION INPUT -------------------------

query = st.text_input("Question:")


# ------------------------- RETRIEVAL + GENERATION -------------------------

if query:

    # Check whether vector database exists
    if os.path.exists(file_path):

        # Load saved FAISS vector store
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

        # Create retriever
        retriever = vectorstore.as_retriever()

        # Convert retrieved documents into a single string
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Prompt Template
        prompt = ChatPromptTemplate.from_template("""
                    You are a helpful assistant.

                    Use only the context below to answer the user's question.

                    If the answer is not present in the context, simply say:
                    "I don't know."

                    Context:
                    {context}

                    Question:
                    {input}
                    """)

        # RAG Pipeline
        chain = (
            {
                "context": retriever | format_docs,
                "input": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        # Generate Answer
        result = chain.invoke(query)

        st.header("Answer")
        st.write(result)

    else:
        st.warning("Please process the URLs first.")