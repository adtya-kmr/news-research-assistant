from dotenv import load_dotenv
import os

import streamlit as st
import pickle
import langchain
from langchain_core.tracers import ConsoleCallbackHandler
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm_endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    temperature=0.9
)

llm = ChatHuggingFace(llm=llm_endpoint)

st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

file_path = "faiss_store_hfai.pkl"
process_url_clicked = st.sidebar.button("Process URLs")

main_placeholder = st.empty()

if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    main_placeholder.text("Data Loading...Started...✅✅✅")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")

    docs = text_splitter.split_documents(data)
    embeddings = HuggingFaceEmbeddings()
    vectorstore_hfai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")

    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_hfai, f)

query = main_placeholder.text_input("Question: ")

if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

        retriever = vectorstore.as_retriever()
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant. Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: {context}
        Question: {input}
        """)
        chain = (
                {"context": retriever | format_docs, "input": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
        )
        result = chain.invoke(query)
        st.header("Answer")
        st.write(result)