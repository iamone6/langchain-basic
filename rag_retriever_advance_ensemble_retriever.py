#
#   Sparse Retriever : 키워드(용어) 매칭 기반 검색기입니다. TF-IDF, BM25 등이 대표적이며,
#       문서를 "대부분의 값이 0인 고차원 벡터(sparse vector)"로 표현해 단어의 등장 여부/빈도로 유사도를 계산합니다.
#       정확한 키워드나 고유명사, 숫자처럼 의미 임베딩으로는 잘 안 잡히는 단어 매칭에 강합니다.
#       (ex: langchain_community.retrievers 의 BM25Retriever, rank_bm25 패키지 필요 poetry add rank_bm25)
#
#   Dense Retriever : 의미(semantic) 기반 검색기입니다. 이 프로젝트에서 지금까지 써온
#       HuggingFaceEmbeddings + Chroma/FAISS 방식이 여기 해당하며, 문서를 "모든 차원에 값이 있는
#       저차원 벡터(dense vector)"로 임베딩해 의미적으로 유사한 문서를 찾습니다.
#       표현이 달라도 의미가 비슷하면 찾아내지만, 정확한 키워드 매칭에는 약할 수 있습니다.
#
#   Ensemble Retriever : 위 Sparse Retriever 와 Dense Retriever 를 함께 사용하여,
#       각각의 검색 결과를 weights 를 적용한 순위 결합(RRF, Reciprocal Rank Fusion 등)으로 합쳐
#       하나의 순위로 반환하는 retriever 입니다. 키워드 매칭의 정확성과 의미 검색의 유연함을 동시에 취할 수 있습니다.
#       (langchain_classic.retrievers.ensemble 의 EnsembleRetriever)
#
#   아래는 이 Ensemble Retriever 에 대한 예제 코드입니다.
#       rag_vectorstore.py 가 저장해둔 ./chroma_db 를 dense 쪽 retriever 로 그대로 재사용합니다.
#       BM25Retriever(sparse)는 vectorstore 가 아니라 원문 텍스트 리스트로 자체 인덱스를 만들기 때문에,
#       chromaDB.get() 으로 저장된 청크 원문을 꺼내와 사용합니다. (id 없이 여러 번 저장해 중복된 청크가 섞여있어 dedupe 필요)
#       BM25 의 기본 preprocess_func 는 공백 기준 토큰화라 한국어 형태소 분석 없이 그대로 씁니다(완벽하지 않음).
#
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
from langchain_chroma.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from utils import embeddings
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

claude = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)

# dense retriever : rag_vectorstore.py 가 저장해둔 vector store 를 그대로 로드
chromaDB = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
dense_retriever = chromaDB.as_retriever(search_kwargs={"k": 3})

# sparse retriever : chromaDB 에 저장된 청크 원문을 꺼내와 BM25 인덱스를 새로 구성
# (id 지정 없이 여러 번 저장된 이력이 있어 동일 청크가 중복 저장돼 있으므로 dedupe)
stored_chunks = list(set(chromaDB.get()["documents"]))
sparse_retriever = BM25Retriever.from_texts(stored_chunks)
sparse_retriever.k = 3

# 두 retriever 의 결과를 weights 비율로 결합 (합이 1일 필요는 없지만 관례상 1로 맞춤)
ensemble_retriever = EnsembleRetriever(retrievers=[sparse_retriever, dense_retriever], weights=[0.5, 0.5])

if __name__ == "__main__":
    query = "저비스 펜들턴"  # 고유명사 -> BM25(키워드)가 강점을 보일 만한 질문

    print(f"Q: {query}\n")

    print("[BM25 (sparse) 검색 결과]")
    for doc in sparse_retriever.invoke(query):
        print(f"  - {doc.page_content[:80]}")

    print("\n[Chroma (dense) 검색 결과]")
    for doc in dense_retriever.invoke(query):
        print(f"  - {doc.page_content[:80]}")

    print("\n[EnsembleRetriever 검색 결과] (둘을 결합한 순위)")
    ensemble_docs = ensemble_retriever.invoke(query)
    for doc in ensemble_docs:
        print(f"  - {doc.page_content[:80]}")

    # Stuff 방식 : EnsembleRetriever 로 검색된 청크 전체를 컨텍스트로 한 번에 LLM에 전달하여 답변 생성
    prompt = ChatPromptTemplate.from_messages([
        ("system", "다음 컨텍스트를 참고하여 질문에 답변해줘.\n\n{context}"),
        ("human", "{question}"),
    ])
    chain = prompt | claude | StrOutputParser()
    context = "\n\n".join(doc.page_content for doc in ensemble_docs)
    answer = chain.invoke({"context": context, "question": query})

    print(f"\nA: {answer}")
