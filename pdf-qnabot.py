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
st.title("PDF 기반 Q&A 챗봇")

# OpenAI API 키 설정
openai_api_key = st.secrets["openai_api_key"]
os.environ["OPENAI_API_KEY"] = openai_api_key

# 세션 상태 초기화
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []
if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = None
if "user_question" not in st.session_state:
    st.session_state.user_question = ""

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
    user_question = st.text_input("질문을 입력하세요:", value=st.session_state.user_question, key="user_input")
    if user_question:
        docs = st.session_state.knowledge_base.similarity_search(user_question)
        
        # temperature 설정
        temperature = temperature_mapping[temperature_option]
        
        # LLM 설정
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=temperature)
        
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=user_question)
        st.session_state.qa_history.append({"question": user_question, "answer": response})
        
        # 입력 필드 초기화
        st.session_state.user_question = ""
        st.experimental_rerun()

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

# 현재 사용 중인 설정 표시
st.sidebar.write(f"현재 사용 중인 모델: gpt-4o-mini")
st.sidebar.write(f"현재 응답 스타일: {temperature_option}")
