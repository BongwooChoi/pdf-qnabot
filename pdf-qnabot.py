import os
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from PyPDF2 import PdfReader

# Streamlit 앱 설정
st.set_page_config(page_title="PDF Q&A 챗봇", layout="wide")

# 사이드바 설정
st.sidebar.title("설정")
pdfs = st.sidebar.file_uploader("PDF 파일을 업로드하세요", type="pdf", accept_multiple_files=True)

# 모델 선택 옵션 추가
model_option = st.sidebar.selectbox(
    "모델을 선택하세요",
    ("gpt-4o-mini", "gpt-3.5-turbo")
)

temperature_option = st.sidebar.selectbox(
    "응답 스타일을 선택하세요",
    ("일관적인 (0)", "균형잡힌 (0.5)", "창의적인 (1)")
)

# temperature 값 매핑
temperature_mapping = {
    "일관적인 (0)": 0,
    "균형잡힌 (0.5)": 0.5,
    "창의적인 (1)": 1
}

# 대화 내역 초기화 버튼
if st.sidebar.button("대화 내역 초기화"):
    st.session_state.qa_history = []
    st.sidebar.success("대화 내역이 초기화되었습니다.")

# 메인 화면 설정
st.title("RAG 기반 PDF Q&A 챗봇")
st.subheader("업로드한 PDF 문서를 기반으로 답변하는 챗봇입니다.")
st.markdown("※ RAG(Retrieval Augmented Generation): 답변 시 벡터DB에서 문서 내용을 검색하여 더 정확한 답변 생성")

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

# 기본 PDF 파일 로드 및 처리
def load_default_pdf():
    default_pdf_path = "data/default.pdf"
    if os.path.exists(default_pdf_path):
        with open(default_pdf_path, "rb") as pdf_file:
            process_pdfs([pdf_file])
        st.sidebar.info("기본 PDF 파일이 로드되었습니다.")
    else:
        st.sidebar.warning("기본 PDF 파일을 찾을 수 없습니다.")

# PDF 업로드 또는 기본 PDF 로드
if pdfs:
    process_pdfs(pdfs)
elif st.session_state.knowledge_base is None:
    load_default_pdf()

# 챗봇 인터페이스
st.write("---")
if st.session_state.knowledge_base is not None:
    user_question = st.text_area("질문을 입력하세요:", height=100)
    if st.button("질문하기"):
        if user_question:
            docs = st.session_state.knowledge_base.similarity_search(user_question)
            
            # temperature 설정
            temperature = temperature_mapping[temperature_option]
            
            # LLM 설정
            llm = ChatOpenAI(model_name=model_option, temperature=temperature)
            
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
    st.info("PDF 파일을 업로드하거나 기본 PDF 파일이 로드될 때까지 기다려주세요.")

# 현재 사용 중인 설정 표시
st.sidebar.write(f"현재 사용 중인 모델: {model_option}")
st.sidebar.write(f"현재 응답 스타일: {temperature_option}")
