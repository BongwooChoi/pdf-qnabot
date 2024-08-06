import os
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from PyPDF2 import PdfReader

# Streamlit 앱 설정
st.set_page_config(page_title="PDF 기반 Q&A 챗봇", layout="wide")

# 사이드바 설정
st.sidebar.title("설정")
pdfs = st.sidebar.file_uploader("PDF 파일을 업로드하세요", type="pdf", accept_multiple_files=True)
model_option = st.sidebar.selectbox(
    "사용할 모델을 선택하세요",
    ("GPT-4o-mini", "GPT-3.5-turbo")
)

# 메인 화면 설정
st.title("PDF 기반 Q&A 챗봇")

# OpenAI API 키 설정
openai_api_key = st.secrets["openai_api_key"]
os.environ["OPENAI_API_KEY"] = openai_api_key

# 세션 상태 초기화
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []
if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = None

# PDF 처리 함수
def process_pdfs(pdf_files):
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    embeddings = OpenAIEmbeddings()
    st.session_state.knowledge_base = FAISS.from_texts(chunks, embeddings)
    st.sidebar.success(f"{len(pdf_files)}개의 PDF가 성공적으로 처리되었습니다!")

# PDF 업로드 시 처리
if pdfs and st.session_state.knowledge_base is None:
    process_pdfs(pdfs)
       
# 챗봇 인터페이스
st.write("---")
if st.session_state.knowledge_base is not None:
    user_question = st.text_input("질문을 입력하세요:")
    if user_question:
        docs = st.session_state.knowledge_base.similarity_search(user_question)
        
        # 선택된 모델에 따라 LLM 설정
        if model_option == "GPT-4o-mini":
            llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        else:
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=user_question)
        st.session_state.qa_history.append({"question": user_question, "answer": response})

    # 채팅 기록 표시
    for qa in reversed(st.session_state.qa_history):
        message_container = st.container()
        with message_container:
            col1, col2 = st.columns([1, 9])
            with col1:
                st.image("https://via.placeholder.com/40x40.png?text=You", width=40)
            with col2:
                st.markdown(f"**You:** {qa['question']}")
            
            col1, col2 = st.columns([1, 9])
            with col1:
                st.image("https://via.placeholder.com/40x40.png?text=Bot", width=40)
            with col2:
                st.markdown(f"**Bot:** {qa['answer']}")
        st.write("---")
else:
    st.info("좌측 사이드바에서 PDF 파일들을 업로드해주세요.")

# 현재 사용 중인 모델 표시
st.sidebar.write(f"현재 사용 중인 모델: {model_option}")
