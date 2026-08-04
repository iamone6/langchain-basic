# LangChain 기본 학습 프로젝트

**LangChain과 Anthropic Claude API를 활용한 실무형 RAG(Retrieval-Augmented Generation) 및 LLM 에이전트 학습 프로젝트입니다.**

## 프로젝트 개요

이 프로젝트는 LangChain 1.x 버전을 기반으로 다음 내용들을 실습합니다:
- Claude API를 통한 LLM 기본 호출 및 프롬프트 엔지니어링
- 문서 로딩 및 다양한 청킹 기법
- 임베딩 및 벡터스토어 구축
- RAG 파이프라인의 4가지 컨텍스트 체인 구현
- 검색 성능 향상을 위한 Retriever 고급 기법
- 자연언어 SQL 쿼리를 위한 에이전트 개발

## 기술 스택

- **LLM**: Anthropic Claude 3.5 Sonnet via `langchain_anthropic`
- **벡터스토어**: Chroma, FAISS
- **임베딩**: HuggingFace (`jhgan/ko-sroberta-multitask`)
- **문서처리**: PyPDF, python-docx, openpyxl, BeautifulSoup4
- **데이터베이스**: SQLite
- **패키지관리**: Poetry, Python 3.14, `.venv` 가상환경

## 실행 환경

```bash
# 패키지 설치
poetry install

# 실행
poetry run python main.py
```

### 환경 변수 (.env)
```
ANTHROPIC_API_KEY=sk-ant-...
```

---

## 학습 파일 구성

### 1️⃣ **LangChain 기본** (`main.py`)

**목표**: LangChain의 핵심 API와 LCEL(LangChain Expression Language) 마스터

**학습 내용**:
- `ChatAnthropic` — Claude 모델 호출
- `ChatPromptTemplate` — 시스템/유저 메시지 템플릿 구성
- `FewShotChatMessagePromptTemplate` — few-shot 프롬프트 엔지니어링
- `StrOutputParser` / `JsonOutputParser` — 출력 형식 변환
- LCEL 체인 구성: `prompt | claude | parser`
- `.invoke()` vs `.stream()` — 단일/스트리밍 응답

---

### 2️⃣ **RAG 문서 로딩** (`rag-document-loaders.py`, `rag_load_various_file_format.py`)

**목표**: 다양한 파일 형식을 문서로 변환하는 방법 숙달

**지원 포맷 및 기법**:
- **TXT**: `open()` + `Document` 래핑
- **PDF**: `PyPDFLoader.load_and_split()` — 페이지 단위 분리
- **DOCX**: `Docx2txtLoader` — 전체 문서를 단일 Document로 변환
- **XLSX**: `openpyxl` + 직접 텍스트 변환 (전용 로더 부재)
- **WEB**: `WebBaseLoader` with `USER_AGENT` 설정

**핵심 개념**:
- RAG 파이프라인: `Loaders → Splitters → Embeddings → VectorStores → Retrieval`
- LangChain Document 객체 구조 및 메타데이터 활용

---

### 3️⃣ **RAG 청킹 기법** (`rag_chunking.py`)

**목표**: 텍스트 분할 방식의 장단점 이해 및 선택

**구현 방식**:
| 방식 | 특징 | 사용 시기 |
|------|------|---------|
| **CharacterTextSplitter** | 단일 구분자로 분할 | 단순한 텍스트 문서 |
| **RecursiveCharacterTextSplitter** | 여러 구분자 순차 적용 (문단→줄→문장→단어) | 일반적인 프로덕션 환경 |
| **TokenTextSplitter** | 토큰 수 기준 분할 | LLM 토큰 한계 관리 필요시 |

**학습 포인트**:
- `length_function` 지정으로 글자 vs 토큰 기준 선택
- `split_text()` vs `split_documents()`
- `Language` enum을 통한 코드/HTML 특화 분할

---

### 4️⃣ **RAG 청킹 고급 기법** (`rag_chunking_advance.py`)

**목표**: 데이터 유형별 최적화된 청킹 전략 습득

**4가지 고급 기법**:

1. **코드 전문화 청킹**
   - `RecursiveCharacterTextSplitter.from_language(Language.PYTHON, ...)`
   - 클래스/함수 경계를 우선 존중하여 시멘틱 무결성 보장

2. **마크다운 헤더 기반 청킹**
   - `MarkdownHeaderTextSplitter`
   - `#`/`##`/`###` 계층 구조 추적
   - 각 청크 metadata에 헤더 경로 자동 기록

3. **의미론적 청킹 (Semantic Chunking)**
   - `langchain_experimental.SemanticChunker`
   - 문장 임베딩으로 의미 변화 지점 감지
   - 고정 길이 분할보다 문맥 보존도 높음 (성능 대비 비용 증가)

4. **표준 재귀 청킹**
   - 구분자 계층 자동 적용으로 일반적 우수 성능

---

### 5️⃣ **RAG 임베딩** (`rag_embedding.py`)

**목표**: 벡터 표현을 통한 의미론적 검색 구현

**구현 내용**:
- HuggingFace 무료 모델 활용: `jhgan/ko-sroberta-multitask`
- `HuggingFaceEmbeddings` 설정 (디바이스, 정규화)
- `embed_documents()` vs `embed_query()` 구분 호출
- 코사인 유사도 직접 계산: `dot(a, b) / (norm(a) * norm(b))`

**핵심 개념**:
- Claude는 임베딩 API 미지원 → 오픈소스 모델 대체
- 한국어 특화 모델 선정의 중요성

---

### 6️⃣ **RAG 벡터스토어** (`rag_vectorstore.py`)

**목표**: 임베딩된 문서의 저장 및 검색 최적화

**구현 벡터스토어**:

| DB | 저장 | 검색 | 특징 |
|----|------|------|------|
| **Chroma** | `from_texts()` + `persist_directory` | `similarity_search()` | 실메모리/디스크 선택 가능 |
| **FAISS** | `save_local()` | `load_local()` | 대규모 빠른 검색 |

**검색 전략**:
- `similarity_search()` — 코사인 유사도 기반
- `similarity_search_with_score()` — 점수 포함 반환
- `max_marginal_relevance_search()` (MMR) — 유사도 + 다양성 균형

**학습 포인트**:
- 문서 ID 관리의 중요성 (update/delete 필요시 ID 사전 지정)
- vectorstore vs docstore 역할 분담

---

### 7️⃣ **RAG 컨텍스트 체인** (`rag_retrieval.py`)

**목표**: 검색 결과를 활용한 4가지 답변 생성 방식 비교

**4가지 체인 유형** (LangChain 1.x LCEL로 직접 구현):

1. **Stuff**
   - 검색된 모든 청크를 한 번에 컨텍스트로 전달
   - 장점: 단순, 빠름 | 단점: 토큰 한계, 컨텍스트 윈도우 제약

2. **Map Reduce**
   - 청크마다 병렬로 정보 추출(MAP) → 종합 처리(REDUCE)
   - 장점: 대량 문서 처리 | 단점: 청크 간 종합 손실 가능

3. **Refine**
   - 청크를 순차 처리하며 이전 답변 개선
   - 장점: 누적 학습 | 단점: 시간 소요

4. **Map Rerank**
   - 청크마다 (answer, score) 생성 후 최고 점수 선택
   - 장점: 신뢰도 평가 | 단점: 완성도 감소 가능

**구현 방식**:
- `enums/chain_type.py`의 `ChainType` enum으로 선택
- `Retrieval` 클래스가 chain_type에 따라 메서드 분기

---

### 8️⃣ **Retriever 고급 기법** (`rag_retriever_advance_*.py`)

**목표**: 검색 정확도와 효율성 극대화

**5가지 고급 기법**:

**1. MultiQueryRetriever** — 질문 다양화
- 원래 질문을 LLM으로 3개 버전으로 확장
- 각각 별도 검색 → 중복 제거한 합집합 반환
- 약자 vs 풀네임, 다양한 표현 커버

**2. ParentDocumentRetriever** — 계층적 검색
- `child_splitter`: 작은 청크로 정확도 추구
- `parent_splitter`: 큰 청크로 풍부한 컨텍스트
- vectorstore는 child만 저장, docstore는 parent 보관
- 검색 시: 유사한 child 찾기 → parent 조회

**3. SelfQueryRetriever** — 자동 필터링
- 자연언어 질문을 "텍스트 + 구조화 필터"로 분해
- metadata 필드 (저자, 날짜, 카테고리 등) 자동 필터링
- `AttributeInfo`로 필터 가능 필드 미리 정의

**4. TimeWeightedVectorStoreRetriever** — 시간 가중치
- 최근 추가/조회 문서에 높은 가중치
- "Lost in the Middle" 현상 부분 보완
- 동적 데이터에서 최신 정보 우선

**5. EnsembleRetriever** — 하이브리드 검색
- Sparse (키워드, BM25) + Dense (의미, 임베딩) 결합
- 각각의 강점 활용: 정확한 용어 vs 의미 이해
- 개선된 재현율(recall) 달성

**6. LongContextReorder** — "Lost in the Middle" 대응
- 컨텍스트가 길 때 중간 정보 손실 현상 해결
- 관련도 높은 문서를 리스트 앞/뒤로, 낮은 것을 중간으로 배치
- k ≥ 5개의 검색 결과 시 효과적

---

### 9️⃣ **SQL Agent** (`sql_agnet.py`)

**목표**: 자연언어로 SQL 쿼리 생성 및 실행

**구현 흐름**:
1. SQLite 데이터베이스 연결 (`SQLDatabase.from_uri()`)
2. `SQLDatabaseToolkit`으로 스키마 인식
3. `langchain.agents.create_agent()` — tool-calling agent 생성
4. 동작: 쿼리 생성 → 실행 → 오류 시 자동 수정
5. 최종 결과를 자연언어로 요약

**고급 기능 - Few-shot 벡터 검색**:
- 질문↔SQL 예제 쌍을 Chroma DB에 사전 저장
- 신규 질문 유사도 검색 → 유사 예제 추천
- 임계값 이상 유사도면 참고 쿼리를 prompt에 추가
- 모델의 SQL 생성 정확도 향상

**사용 데이터**:
- `chinook.db` — 음악 스트리밍 서비스 샘플 데이터베이스

---

### 🔧 **공용 유틸** (`utils.py`)

**포함 기능**:
- **토큰 카운터**: `tiktoken` 기반 Claude 토큰 추정
- **임베딩 모델**: HuggingFace 모델 싱글톤 객체
- **인코딩**: `cl100k_base` (Claude 근사치)

---

## 주요 학습 성과

✅ **LangChain LCEL** — 체인 구성의 함수형 패러다임 이해  
✅ **RAG 파이프라인** — 문서 로딩부터 최종 답변까지 end-to-end 구현  
✅ **다양한 청킹 기법** — 데이터 유형별 최적화 전략  
✅ **벡터 검색** — 의미론적 유사도 기반 정보 검색  
✅ **고급 Retriever** — 성능 향상을 위한 6가지 기법  
✅ **LLM Agent** — 자연언어 SQL 생성과 자동 오류 수정  
✅ **프롬프트 엔지니어링** — few-shot, 구조화 출력, 역할 지정  

---

## 알려진 이슈 및 해결책

- **DeprecationWarning**: `langchain-community` 단계적 폐기 경고 → `warnings.filterwarnings("ignore")` 처리
- **TimeWeightedVectorStoreRetriever + Chroma**: Chroma는 datetime 메타데이터 미지원 → FAISS 사용
- **SelfQueryRetriever 번역**: 일부 벡터스토어 translator import 오류 → 명시적으로 `ChromaTranslator()` 지정
- **ParentDocumentRetriever + BM25**: 기존 DB 재사용 시 중복 제거 필요

---

## 참고 자료

- [LangChain 공식 문서](https://python.langchain.com/)
- [Anthropic Claude API](https://docs.anthropic.com/)
- [HuggingFace Embeddings](https://huggingface.co/)
- [Chroma Vector Database](https://docs.trychroma.com/)

---
