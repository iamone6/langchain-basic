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
| `rag_chunking.py` | 문서 청킹 예제 (글자 수 기준, 토큰 수 기준) |
| `rag_embedding.py` | HuggingFace 임베딩 생성 및 코사인 유사도 계산 예제 |
| `rag_vectorstore.py` | Chroma/FAISS 벡터스토어 저장·검색 예제 |
| `rag_retrieval.py` | 컨텍스트 체인 4종(Stuff/Map Reduce/Refine/Map Rerank) 예제 |
| `sql_agnet.py` | SQL Agent + few-shot 벡터 검색 예제 |
| `utils.py` | 공용 유틸(토큰 카운터, 임베딩 모델 객체) |

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

### RAG Chunking (`rag_chunking.py`)

- `CharacterTextSplitter` — 단일 구분자(`separator`)로 텍스트 분할, chunk_size 초과 시 경고
- `RecursiveCharacterTextSplitter` — 여러 구분자를 순차 재귀 적용, 실무에서 일반적으로 사용
- `length_function` — 청크 길이 측정 함수 지정 (`len` 또는 토크나이저)
- `tiktoken` — 오프라인 토크나이저로 토큰 수 기준 청킹 (`cl100k_base` 인코딩, Claude 근사치, `utils.py`에 위치)
- `split_text()` → `list[str]`, `split_documents()` → `list[Document]`
- 코드/HTML/LaTeX 등은 `Language` enum을 추가 지정하여 분할 (ex: `language=Language.PYTHON`)
- `enums/chunk_type.py`의 `ChunkType` enum으로 분할 방식(NORMAL/RECURSIVE/TOKEN) 선택
- 모듈 레벨 실행 코드는 `if __name__ == "__main__":` 으로 감싸 다른 파일에서 import 시 중복 실행 방지

### RAG Embedding (`rag_embedding.py`)

- Claude는 임베딩 전용 API가 없고 OpenAI 임베딩은 유료라서, HuggingFace 무료 모델 사용
- `HuggingFaceEmbeddings` (`langchain_huggingface`) — `model_kwargs={"device": "cpu"}`, `encode_kwargs={"normalize_embeddings": True}`
- 한국어 특화 모델: `jhgan/ko-sroberta-multitask`
- `embed_documents(list[str])` — 문서 청크 임베딩, `embed_query(str)` — 검색 쿼리 임베딩 (내부적으로 다르게 처리될 수 있어 구분 필요)
- 코사인 유사도 직접 계산: `dot(a, b) / (norm(a) * norm(b))`

### RAG VectorStore (`rag_vectorstore.py`)

- `Chroma` (`langchain_chroma`) — `from_texts()`로 생성, `persist_directory` 지정 시 디스크에 저장
- `FAISS` (`langchain_community.vectorstores`) — `save_local()` / `load_local(..., allow_dangerous_deserialization=True)` (pickle 기반이라 직접 생성한 파일에만 사용)
- `similarity_search()` — 유사도 기준 검색, `similarity_search_with_score()` — score 포함 (낮을수록 유사)
- `max_marginal_relevance_search()` (MMR) — 유사도 + 다양성 균형 검색 (`fetch_k`, `lambda_mult` 파라미터)
- ID 미지정 시 UUID 자동 생성 — `update_document()`/`delete()` 등을 쓰려면 저장 시 `ids` 직접 지정 필요
- `langchain-community`의 FAISS는 standalone 패키지가 아직 없어 DeprecationWarning 발생 (동일하게 숨김 처리)

### RAG Retrieval (`rag_retrieval.py`)

컨텍스트 Chain의 4가지 종류 (LangChain 1.x에는 `RetrievalQA`/`chain_type`이 제거되어 LCEL로 직접 구성):

- **Stuff** — 검색된 청크 전체를 컨텍스트로 한 번에 LLM에 전달
- **Map Reduce** — 청크마다 관련 정보를 병렬 추출(MAP) 후 종합해 최종 답변 생성(REDUCE)
- **Refine** — 청크를 순차적으로 돌며 이전 답변을 새 컨텍스트로 계속 개선
- **Map Rerank** — 청크마다 병렬로 (answer, score) 생성 후 최고 score의 answer 채택
- `enums/chain_type.py`의 `ChainType` enum으로 방식 선택, `Retrieval` 클래스가 `chain_type`에 따라 내부 메서드로 분기

### SQL Agent (`sql_agnet.py`)

- `SQLDatabase.from_uri("sqlite:///...")` — SQLite는 서버가 필요 없어 표준 `sqlite3` 모듈로 파일을 직접 읽음
- `SQLDatabaseToolkit` — `list_tables`/`get_schema`/`query_checker`/`run_query` tool 제공 (`langchain-experimental` 불필요)
- `langchain.agents.create_agent` — LangChain 1.x의 tool-calling agent 생성자, `system_prompt`로 역할 지정
- 동작 흐름: 스키마 조회 → 자연어 질문 → SQL 생성 → 실행 → 결과를 자연어로 요약, 에러 시 스스로 재작성 후 재시도
- Few-shot 벡터 검색: 질문↔SQL 예제를 별도 Chroma DB(`sql_fewshot_db`)에 저장해두고, 새 질문이 오면 유사도 검색 → 임계값 이내면 참고 쿼리를 prompt suffix로 추가, 아니면 기존 방식 그대로 유지
- `chinook.db`는 `/home/iamone/src/sqlite/Dockerfile`(데이터 전용 컨테이너)에서 `docker cp`로 로컬 `file/`에 꺼내와 사용
