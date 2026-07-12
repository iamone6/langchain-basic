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
| `rag_load_various_file_format.py` | docx/pdf/xlsx/txt/web url 로딩 예제 (포맷별 로딩 방식 정리) |
| `rag_chunking.py` | 문서 청킹 예제 (글자 수 기준, 토큰 수 기준) |
| `rag_embedding.py` | HuggingFace 임베딩 생성 및 코사인 유사도 계산 예제 |
| `rag_vectorstore.py` | Chroma/FAISS 벡터스토어 저장·검색 예제 |
| `rag_retrieval.py` | 컨텍스트 체인 4종(Stuff/Map Reduce/Refine/Map Rerank) 예제 |
| `rag_retriever_advance_multi_query_retriever.py` | MultiQueryRetriever 예제 (질문을 여러 버전으로 확장해 검색) |
| `rag_retriever_advance_parent_document.py` | ParentDocumentRetriever 예제 (작은 child로 검색, 큰 parent를 컨텍스트로) |
| `rag_retriever_advance_self_query_retriever.py` | SelfQueryRetriever 예제 (질문에서 metadata 필터 자동 추출) |
| `rag_retriever_advance_time_weight_retriever.py` | TimeWeightedVectorStoreRetriever 예제 (최근성 가중치 검색) |
| `rag_retriever_advance_ensemble_retriever.py` | EnsembleRetriever 예제 (Sparse+Dense 결합 검색) |
| `rag_retriever_advance_long_context_reorder.py` | LongContextReorder 예제 (Lost in the Middle 대응) |
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

### RAG 다양한 파일 포맷 로딩 (`rag_load_various_file_format.py`)

- **TXT** — 별도 Loader 불필요, `open()`으로 읽어 `Document`로 감싸기만 하면 됨
- **DOCX** — `Docx2txtLoader.load()`, 워드는 페이지 개념이 없어 전체가 Document 1개로 반환됨
- **PDF** — `PyPDFLoader.load_and_split()`으로 페이지 단위 분리 (`load()`는 전체를 1개로 반환)
- **XLSX** — 전용 Loader 없음, `openpyxl`로 직접 읽어 텍스트 변환 후 `Document`로 감쌈
- **WEB** — `WebBaseLoader`, `os.environ.setdefault("USER_AGENT", ...)`를 **import 전에** 호출해야 경고 안 뜸

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

### RAG Retriever 고급 기법 (`rag_retriever_advance_*.py`)

LangChain 1.x에서 `MultiQueryRetriever`/`ParentDocumentRetriever`/`SelfQueryRetriever`/`TimeWeightedVectorStoreRetriever`는 모두 `langchain_classic` 패키지로 이동됨 (`langchain`에 번들되어 있지 않아 `poetry add langchain-classic` 필요, `poetry-add`로 재설치 시 `tokenizers` 버전이 튈 수 있음 — 아래 "알려진 이슈" 참고)

- **MultiQueryRetriever** — 질문 1개를 LLM으로 여러 버전으로 확장(기본 3개) 후 각각 검색, 중복 제거한 합집합 반환. `generate_queries()`/`retrieve_documents()`/`unique_union()`을 직접 호출하면 `invoke()`와 동일한 결과를 얻으면서 생성된 질문 목록도 확인 가능 (`CallbackManagerForRetrieverRun.get_noop_manager()` 사용)
- **ParentDocumentRetriever** — `child_splitter`(작은 chunk, 검색 정확도용)와 `parent_splitter`(큰 chunk, 컨텍스트용)를 함께 지정. `vectorstore`엔 child만 임베딩되어 저장되고, `docstore`(`InMemoryStore` 등 단순 key-value)엔 parent 원문이 저장됨. child의 metadata에 자동으로 `doc_id`(parent id)가 채워짐. chunking은 `add_documents()` 호출 시 1회만 수행되고, 질문 시점에는 벡터 검색 + id로 parent 조회만 일어남 (재청킹 없음)
- **SelfQueryRetriever** — 질문을 분석해 "의미 검색 텍스트"와 "metadata 필터(구조화 쿼리)"로 분리 후 필터링+검색을 함께 수행. `AttributeInfo`로 필터 가능한 필드(이름/설명/타입)를 미리 정의해야 함. 구조화 쿼리 파싱에 `lark` 패키지 필요 (`poetry add lark`)
- **TimeWeightedVectorStoreRetriever** — `score = (1-decay_rate)^경과시간(hour) + semantic_similarity`로, 최근에 추가/조회된 문서일수록 가중치 부여. `add_documents(docs, current_time=...)`로 문서별 삽입 시점을 과거로 시뮬레이션 가능 (단, 한 번의 호출엔 하나의 `current_time`만 적용되므로 실제 문서마다 다른 시각을 쓰려면 호출 전에 각 `Document.metadata["last_accessed_at"]`을 직접 채워야 함). `other_score_keys`에 임의 metadata 필드(예: `access_count`)를 지정하면 그 값이 점수에 가산되어 "많이 참조된 chunk 가중치" 같은 로직도 구현 가능
- **알려진 이슈**: Chroma는 metadata에 `datetime` 객체 저장 불가(str/int/float/bool/list만 허용) → TimeWeightedVectorStoreRetriever는 `FAISS`(빈 `faiss.IndexFlatL2` + `InMemoryDocstore`)를 사용. `langchain_core.InMemoryVectorStore`는 `_select_relevance_score_fn` 미구현이라 이 retriever와 호환 안 됨. `SelfQueryRetriever.from_llm()`은 translator 자동 감지 시 모든 벡터스토어용 translator를 일괄 import하는데 `langchain-community` 버전에 따라 일부(Databricks) import가 깨질 수 있음 → `structured_query_translator=ChromaTranslator()`로 명시해 우회
- **EnsembleRetriever** — Sparse Retriever(키워드 매칭, `BM25Retriever`, `rank_bm25` 패키지 필요)와 Dense Retriever(의미 임베딩, Chroma 등)를 함께 사용해 결과를 순위 결합. `BM25Retriever`는 vectorstore가 아니라 원문 텍스트 리스트로 자체 인덱스를 구성하므로, 기존 vectorstore를 재사용하려면 `vectorstore.get()`으로 원문을 꺼내와야 함 (`rag_vectorstore.py`가 id 없이 중복 저장한 이력이 있어 dedupe 필요)
- **LongContextReorder** — "Lost in the Middle" 현상(LLM이 긴 컨텍스트의 중간 부분 정보를 잘 놓침) 대응. 관련도 내림차순 문서를 받아 관련도 높은 문서를 리스트의 맨 앞/맨 뒤로 번갈아 재배치하고 낮은 문서를 중간으로 밀어냄. 검색 결과가 많을 때(k=5~10개 이상), 특히 Stuff 방식처럼 전체를 이어붙여 LLM에 전달할 때 효과적

### SQL Agent (`sql_agnet.py`)

- `SQLDatabase.from_uri("sqlite:///...")` — SQLite는 서버가 필요 없어 표준 `sqlite3` 모듈로 파일을 직접 읽음
- `SQLDatabaseToolkit` — `list_tables`/`get_schema`/`query_checker`/`run_query` tool 제공 (`langchain-experimental` 불필요)
- `langchain.agents.create_agent` — LangChain 1.x의 tool-calling agent 생성자, `system_prompt`로 역할 지정
- 동작 흐름: 스키마 조회 → 자연어 질문 → SQL 생성 → 실행 → 결과를 자연어로 요약, 에러 시 스스로 재작성 후 재시도
- Few-shot 벡터 검색: 질문↔SQL 예제를 별도 Chroma DB(`sql_fewshot_db`)에 저장해두고, 새 질문이 오면 유사도 검색 → 임계값 이내면 참고 쿼리를 prompt suffix로 추가, 아니면 기존 방식 그대로 유지
- `chinook.db`는 `/home/iamone/src/sqlite/Dockerfile`(데이터 전용 컨테이너)에서 `docker cp`로 로컬 `file/`에 꺼내와 사용
