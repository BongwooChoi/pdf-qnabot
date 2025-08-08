## pdf-qnabot – PDF 기반 RAG Q&A 챗봇

PDF 문서를 업로드하면 문서 내용을 바탕으로 질문에 답하는 RAG 기반 챗봇입니다. Streamlit UI 위에서 동작하며, 임베딩은 Google Generative AI(Embeddings)와 벡터 검색은 FAISS를 사용합니다. 최종 답변은 Gemini 모델을 통해 생성됩니다.

### 데모
- 배포(예시): [Streamlit 앱](https://kbdfic-pdf-qnabot.streamlit.app/)
- 소스 저장소: [GitHub 리포지토리](https://github.com/BongwooChoi/pdf-qnabot)

### 주요 기능
- **PDF 업로드/기본 로드**: 사이드바에서 PDF 여러 개 업로드 가능. 업로드가 없으면 `data/default.pdf`를 자동 로드(있을 경우).
- **텍스트 분할 및 임베딩**: `CharacterTextSplitter`로 청크 분할 → Google Embeddings(`text-embedding-004`) 생성.
- **벡터 검색(FAISS)**: 유사도 검색을 통해 관련 문서 청크를 조회.
- **Gemini 응답 생성**: `gemini-2.0-flash` 모델 기반 답변 생성. 온도(창의성) 선택 가능.
- **대화 이력 유지**: 세션 단위로 질문/답변 히스토리 표시.
- **안정성 고려**: gRPC 이슈를 피하기 위해 Embeddings/LLM 모두 `transport="rest"` 모드 사용.

### 폴더 구조
```
pdf-qnabot/
├─ data/
│  ├─ default.pdf                # 기본 예시 PDF (선택적으로 로드)
│  └─ default PDF 문서를 업로드해주세요  # 안내 텍스트 파일
├─ pdf-qnabot.py                 # Streamlit 앱 엔트리 포인트
├─ requirements.txt              # 의존성 목록
└─ .streamlit/                   # (로컬에 생성) secrets.toml로 API 키 관리
```

### 기술 스택
- **UI**: Streamlit
- **PDF 파싱**: PyPDF2
- **임베딩/LLM**: `langchain`, `langchain-community`, `langchain-google-genai`, `google-genai`
- **벡터DB**: FAISS (CPU)

### 동작 흐름(RAG Pipeline)
1. 사용자가 PDF를 업로드하거나 기본 PDF를 로드합니다.
2. 문서 텍스트를 추출하고, 길이에 맞게 분할합니다.
3. 각 청크에 대해 Google Embeddings(`text-embedding-004`)를 생성합니다.
4. FAISS 인덱스를 구성하고, 질문 시 유사도 검색으로 관련 청크를 조회합니다.
5. 조회된 컨텍스트를 기반으로 Gemini(`gemini-2.0-flash`)가 답변을 생성합니다.

## 빠른 시작

### 1) 사전 준비
- Python 3.10 이상 권장
- Google API 키 준비 (Google AI for Developers)

#### API 키 설정 방법(택1)
- `.streamlit/secrets.toml` 사용(권장)
  - 파일 경로: 프로젝트 루트에 `.streamlit/secrets.toml` 생성
  - 내용 예시:
    ```toml
    google_api_key = "YOUR_GOOGLE_API_KEY"
    ```
- 환경 변수 사용
  - Windows(PowerShell, 현재 세션만):
    ```powershell
    $env:GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"
    ```
  - Windows(영구, 새 세션):
    ```powershell
    setx GOOGLE_API_KEY "YOUR_GOOGLE_API_KEY"
    ```
  - macOS/Linux:
    ```bash
    export GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
    ```

앱은 다음 우선순위로 키를 읽습니다: `st.secrets["google_api_key"]` → `GOOGLE_API_KEY` → `GEMINI_API_KEY`.

### 2) 의존성 설치
프로젝트 루트에서 다음을 실행하세요.

#### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### macOS/Linux
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3) 실행
```bash
streamlit run pdf-qnabot.py
```

브라우저가 자동으로 열리지 않는 경우 출력된 로컬 URL을 직접 열어주세요.

## 사용 방법
1. 좌측 사이드바에서 PDF 파일을 업로드합니다(여러 개 가능).
2. 모델과 응답 스타일(온도)을 선택합니다.
3. 질문을 입력하고 “질문하기” 버튼을 누르면, 문서 기반 답변이 생성됩니다.
4. 업로드가 없으면 `data/default.pdf`를 자동 로드하려 시도합니다(존재 시).

## 환경 변수/설정
- 모델: 고정 `gemini-2.0-flash`
- 임베딩 모델: `text-embedding-004`
- 청크 설정: `chunk_size=1000`, `chunk_overlap=200`
- 네트워크 전송: `transport="rest"` (Embeddings/LLM 공통)

## 문제 해결(Troubleshooting)
- "Google API Key가 설정되지 않았습니다" 에러
  - `.streamlit/secrets.toml`에 `google_api_key`를 설정하거나, `GOOGLE_API_KEY`/`GEMINI_API_KEY` 환경 변수를 설정하세요.
- PDF 텍스트가 추출되지 않음
  - 스캔본 PDF일 수 있습니다. OCR(예: Tesseract, Cloud Vision API 등)로 텍스트를 추출한 뒤 사용하세요.
- 네트워크/런타임(gRPC) 관련 예외
  - 본 앱은 `transport="rest"`로 구성되어 있어 gRPC 이슈를 회피합니다. 그래도 문제가 지속되면 네트워크/프록시 설정을 확인하세요.

## 배포
### Streamlit Community Cloud
1. 본 저장소를 연결하여 앱을 생성합니다.
2. App URL/브랜치/경로: `pdf-qnabot.py`를 엔트리로 지정합니다.
3. App Secrets에 다음 키를 추가합니다.
   - `google_api_key = "YOUR_GOOGLE_API_KEY"`

## 개발 참고
- 요구 사항: 아래 파일 참고 → `requirements.txt`
- 로컬 비밀키: `.streamlit/secrets.toml` (버전 관리 제외 권장)
- 컨테이너 개발: 리포지토리에 `.devcontainer/` 구성이 있을 수 있습니다(환경에 따라 표시되지 않을 수 있음).

## 라이선스
해당 리포지토리의 라이선스는 현재 명시되어 있지 않습니다. 필요 시 `LICENSE` 파일을 추가해 주세요.

## 참고 링크
- GitHub 리포지토리: [BongwooChoi/pdf-qnabot](https://github.com/BongwooChoi/pdf-qnabot)
- 배포 예시: [Streamlit 앱](https://kbdfic-pdf-qnabot.streamlit.app/)


