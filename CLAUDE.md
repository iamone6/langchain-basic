# langchain-basic

LangChain 기본 사용법을 학습하기 위한 예제 코드 프로젝트입니다.

## 프로젝트 개요

- LangChain + Anthropic Claude API 연동 예제
- Poetry로 패키지 관리, Python 3.14, `.venv` 가상환경 사용

## 규칙

- git commit 메시지는 한글로 작성

## 실행

```bash
poetry run python main.py
```

## 환경 변수

`.env` 파일에 Anthropic API 키가 필요합니다:

```
ANTHROPIC_API_KEY=sk-ant-...
```

## 예제 파일

| 파일 | 내용 |
|---|---|
| `main.py` | Claude API 기본 호출, 프롬프트 템플릿, 스트리밍 |
| `rag-document-loaders.py` | 다양한 포맷의 문서 로딩 예제 |
| `rag-chunking.py` | 문서 청킹 예제 (글자 수 기준, 토큰 수 기준) |

## 학습 내용

### LangChain 기본 (`main.py`)

- `ChatAnthropic` — Claude 모델 호출 (`langchain_anthropic`)
- `ChatPromptTemplate` — 시스템/유저 메시지 템플릿 구성
- `FewShotChatMessagePromptTemplate` — few-shot 예시 삽입
- `StrOutputParser` / `JsonOutputParser` — 출력 포맷 변환
- LCEL (`prompt | claude | parser`) — 체인 구성
- `.invoke()` — 단일 응답
- `.stream()` — 스트리밍 응답

### RAG Document Loaders (`rag-document-loaders.py`)

RAG 파이프라인: `DocumentLoaders → TextSplitters → Embedding → VectorStores → RetrievalQA`

- `WebBaseLoader` — 웹 페이지 로드 (`beautifulsoup4` 필요)
- `PyPDFLoader` — PDF 로드, `load_and_split()`으로 페이지 단위 분리 (`pypdf` 필요)
- `Docx2txtLoader` — DOCX 로드 (`docx2txt` 필요)
- `openpyxl` + `Document` — XLSX 로드 (`unstructured` Python 3.14 미지원으로 직접 구현)
- `CSVLoader` — CSV 로드, 행마다 Document 객체 반환

> `langchain-community` sunset 예고로 DeprecationWarning 발생 — `warnings.filterwarnings("ignore")` 로 숨김

### RAG Chunking (`rag-chunking.py`)

- `CharacterTextSplitter` — 단일 구분자(`separator`)로 텍스트 분할, chunk_size 초과 시 경고
- `RecursiveCharacterTextSplitter` — 여러 구분자를 순차 재귀 적용, 실무에서 일반적으로 사용
- `length_function` — 청크 길이 측정 함수 지정 (`len` 또는 토크나이저)
- `tiktoken` — 오프라인 토크나이저로 토큰 수 기준 청킹 (`cl100k_base` 인코딩, Claude 근사치)
- `split_text()` → `list[str]`, `split_documents()` → `list[Document]`
- 코드/HTML/LaTeX 등은 `Language` enum을 추가 지정하여 분할 (ex: `language=Language.PYTHON`)
