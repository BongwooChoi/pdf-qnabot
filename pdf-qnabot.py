# pdf-qnabot.py
import os
import asyncio
import streamlit as st
from PyPDF2 import PdfReader

from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

# =========================
# 기본 설정
# =========================
st.set_page_config(page_title="RAG 기반 Q&A 챗봇", page_icon="🤖", layout="wide")

# --- Streamlit 실행 스레드에서 이벤트 루프 보장 (gRPC/AIO 이슈 회피) ---
def _ensure_event_loop():
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

_ensure_event_loop()

# =========================
# 사이드바 UI
# =========================
st.sidebar.title("설정")
pdfs = st.sidebar.file_uploader("PDF 파일을 업로드하세요", type="pdf", accept_multiple_files=True)

# 모델 고정 (Gemini)
MODEL_NAME = "gemini-2.0-flash"
model_option = st.sidebar.selectbox("모델을 선택하세요", (MODEL_NAME,), index=0)

temperature_option = st.sidebar.selectbox(
    "응답 스타일을 선택하세요",
    ("일관적인 (0)", "균형잡힌 (0.5)", "창의적인 (1)")
)

temperature_mapping = {
    "일관적인 (0)": 0,
    "균형잡힌 (0.5)": 0.5,
    "창의적인 (1)": 1
}

# 대화 내역 초기화
if st.sidebar.button("대화 내역 초기화"):
    st.session_state.qa_history = []
    st.sidebar.success("대화 내역이 초기화되었습니다.")

# =========================
# 메인 헤더
# =========================
st.title("RAG 기반 Q&A 챗봇🤖")
st.subheader("업로드한 PDF 문서📋 내용을 바탕으로 답변하는 챗봇입니다.")
st.markdown("※ RAG(Retrieval Augmented Generation): 문서 임베딩 → 벡터DB 검색 → 컨텍스트로 답변 생성")

# =========================
# API 키 설정
# =========================
google_api_key = st.secrets.get("google_api_key", os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
if not google_api_key:
    st.error("Google API Key가 설정되지 않았습니다. `.streamlit/secrets.toml`의 `google_api_key` 또는 환경변수 `GOOGLE_API_KEY`/`GEMINI_API_KEY`를 설정하세요.")
    st.stop()
os.environ["GOOGLE_API_KEY"] = google_api_key

# =========================
# 세션 상태
# =========================
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []
if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = None

# =========================
# 유틸 함수
# =========================
def process_pdfs(pdf_files):
    """PDF들을 읽어 텍스트를 추출하고, 임베딩 후 FAISS 인덱스를 세션에 저장"""
    text = ""
    for pdf in pdf_files:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text() or ""
                text += page_text
        except Exception as e:
            st.warning(f"PDF 처리 중 오류 발생: {e}")

    if not text.strip():
        st.warning("PDF에서 텍스트를 추출하지 못했습니다. 스캔본 PDF일 수 있습니다(OCR 필요).")
        return

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # 이벤트 루프 보장 + REST 전송으로 초기화
    _ensure_event_loop()
    embeddings = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",  # 임베딩 모델
        transport="rest",            # gRPC 대신 REST를 사용해 런타임 이슈 회피
    )

    st.session_state.knowledge_base = FAISS.from_texts(chunks, embeddings)
    st.sidebar.success(f"{len(pdf_files)}개의 PDF가 성공적으로 처리되었습니다!")

def load_default_pdf():
    """기본 PDF를 data/default.pdf에서 로드(선택 기능)"""
    default_pdf_path = "data/default.pdf"
    if os.path.exists(default_pdf_path):
        with open(default_pdf_path, "rb") as pdf_file:
            process_pdfs([pdf_file])
        st.sidebar.info("기본 PDF 파일이 로드되었습니다.")
    else:
        st.sidebar.info("기본 PDF 파일이 없습니다. PDF 문서를 업로드해주세요.")

# =========================
# PDF 업로드/로드
# =========================
if pdfs:
    process_pdfs(pdfs)
elif st.session_state.knowledge_base is None:
    load_default_pdf()

# =========================
# 챗봇 인터페이스
# =========================
st.write("---")
if st.session_state.knowledge_base is not None:
    user_question = st.text_area("질문을 입력하세요:", height=100, placeholder="문서와 관련된 질문을 입력해 보세요.")
    ask = st.button("질문하기")

    if ask and user_question:
        # 검색
        docs = st.session_state.knowledge_base.similarity_search(user_question, k=4)

        # LLM 설정
        temperature = temperature_mapping[temperature_option]
        _ensure_event_loop()
        llm = ChatGoogleGenerativeAI(
            model=model_option,         # "gemini-2.0-flash"
            temperature=temperature,
            transport="rest",           # 동일하게 REST 사용(안전)
        )

        # QA 체인
        chain = load_qa_chain(llm, chain_type="stuff")
        try:
            response = chain.run(input_documents=docs, question=user_question)
        except Exception as e:
            st.error(f"응답 생성 중 오류가 발생했습니다: {e}")
            response = "죄송합니다. 응답 생성 중 문제가 발생했어요."

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
    st.info("좌측 사이드바에서 PDF 문서를 업로드해주세요. 업로드된 PDF가 없어 질문을 할 수 없습니다.")
    st.text_area("질문을 입력하세요:", height=100, disabled=True)
    st.button("질문하기", disabled=True)

# =========================
# 현재 설정 표시
# =========================
st.sidebar.write(f"현재 사용 중인 모델: {model_option}")
st.sidebar.write(f"현재 응답 스타일: {temperature_option}")
