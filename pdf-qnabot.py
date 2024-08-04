# -*- coding: utf-8 -*-
"""pdf-qnabot

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1fj1PtGdnnAip1fMVXpCiMv_BVrS6hKEm
"""

#!pip install streamlit langchain langchain_community openai PyPDF2 faiss-cpu tiktoken

import os
import streamlit as st
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import io
import tempfile

# OpenAI API 키 가져오기
openai_api_key = st.secrets["openai_api_key"]
os.environ["OPENAI_API_KEY"] = openai_api_key

def load_document(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name
    # 임시 파일의 디렉토리 경로 가져오기
    temp_dir = os.path.dirname(temp_file_path)
    # DirectoryLoader에 디렉토리 경로 전달
    loader = DirectoryLoader(temp_dir, glob="*.pdf")
    documents = loader.load()
    
def create_vector_db(docs):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

st.title("PDF 기반 Q&A 챗봇")

uploaded_file = st.file_uploader("PDF 파일 업로드", type=["pdf"])

if uploaded_file is not None:
    file_bytes = io.BytesIO(uploaded_file.read())
    docs = load_document(file_bytes)  
    vectorstore = create_vector_db(docs)
    qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(), chain_type="stuff", retriever=vectorstore.as_retriever())
    
    # 이전 질문 및 답변 저장
    if "history" not in st.session_state:
        st.session_state.history = []

    user_question = st.text_input("질문을 입력하세요:")
    if user_question:
        answer = qa_chain.run(user_question)
        st.session_state.history.append((user_question, answer))
    
    # 이전 질문 및 답변 표시
    if st.session_state.history:
        st.subheader("이전 질문 및 답변:")
        for question, answer in st.session_state.history:
            st.write(f"**질문:** {question}")
            st.write(f"**답변:** {answer}")
