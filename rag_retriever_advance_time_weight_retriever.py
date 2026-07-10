#
#   Retriever 의 4가지 고급 기법에 대해 설명하고, 각 기법에 대한 예시를 제공하는 코드
#   1. multi query retriever : 대충 질문해도 관련된 chunk 를 찾을 수 있도록 
#       여러 유사 질문을 생성하여 찾은 chunk 를 LLM 에 context 로 추가하여 
#       원본 query 와 함께 LLM 에 질문하는 retriever
#   2. parent document retriever : 앞뒤 문맥을 잘 담기 위해 chunk 가 속한 parent document 를 찾아주는 retriever
#       parent document 란, 검색된 chunk 가 속한 원본 document 를 의미하는데, 
#       parent 를 chunking 하지 않았다면 원본 문서 전체가 될 것이고, parent 를 chunking 했다면 parent document 는 chunking 된 상위 chunk 가 될 것입니다.(문서의 page 단위가 아님)
#       paarent document 를 chunking 하려면 Document 객체를 생성할 때 metadata 에 parent_id 를 지정하고, chunking 시 parent_id 를 상위 chunk 의 id 로 지정하면 됩니다.
#   3. self query retriever : 정확한 값을 위해 질문을 분석하여 keyword 를 추출하고, keyword 를 기반으로 vector similarity 를 수행하는 retriever
#   4. time-weighted retriever : 최신 자료를 참고하기 위한 vector similarity + timestamp 를 결합한 retriever
#
#  아래는 위 4가지 retriever 를 결합한 고급 retriever 기법에 대한 설명
#   5 hybrid retriever : vector similarity + keyword 를 결합한 retriever
#   6. semantic hybrid retriever : vector similarity + keyword + semantic search 를 결합한 retriever
#   7. time-weighted hybrid retriever : vector similarity + keyword + timestamp 를 결합한 retriever
#   8. time-weighted semantic hybrid retriever : vector similarity + keyword + timestamp + semantic search 를 결합한 retriever
#   9. time-weighted semantic hybrid retriever with multi query : vector similarity + keyword + timestamp + semantic search + multi query 를 결합한 retriever
#
#   4. TimeWeightedVectorStoreRetriever 예제
#       score = (1 - decay_rate)^(경과 시간(hour)) + semantic similarity 로 계산하여,
#       의미상 비슷하더라도 최근에 추가/조회된 문서일수록 점수가 높아지도록 만드는 retriever입니다.
#       add_documents(docs, current_time=...) 로 문서가 추가된 시점을 과거로 지정할 수 있어,
#       "몇 주 전에 쓴, 의미가 더 비슷한 메모"보다 "방금 쓴, 의미가 덜 비슷한 메모"가 더 높은 점수로
#       검색되는 걸 보여줄 수 있습니다.
#       self query retriever 예제의 소설 데이터(year)는 "출간연도"라서 의미가 다르므로 재사용하지 않고,
#       시점을 다르게 하여 추가한 메모 Document 를 새로 만들어 사용합니다.
#
import datetime
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import faiss
from dotenv import load_dotenv
from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers.time_weighted_retriever import TimeWeightedVectorStoreRetriever
from langchain_core.documents import Document
from utils import embeddings

load_dotenv()

now = datetime.datetime.now()

# TimeWeightedVectorStoreRetriever 는 문서 metadata 에 created_at/last_accessed_at 을
# datetime 객체 그대로 저장하는데, Chroma 는 metadata 로 str/int/float/bool/list 만 허용해서 datetime 을 못 받고,
# langchain_core 의 InMemoryVectorStore 는 relevance score 변환 함수가 구현되어 있지 않아 이 retriever 와 함께 쓸 수 없다.
# 그래서 이 retriever 는 순수 파이썬 객체를 그대로 담아두면서 relevance score 변환도 기본 제공하는 FAISS 를 빈 인덱스로 만들어 사용한다.
# TimeWeightedVectorStoreRetriever 는 내부적으로 memory_stream(문서 리스트)도 함께 관리하므로,
# 문서는 vectorstore 가 아니라 retriever.add_documents() 를 통해 넣어야 last_accessed_at/created_at 이 채워진다.
embedding_dim = len(embeddings.embed_query("dimension probe"))
memo_vectorstore = FAISS(
    embedding_function=embeddings,
    index=faiss.IndexFlatL2(embedding_dim),
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

# decay_rate 가 클수록 시간에 따른 점수 하락이 빨라져 "최근성"이 더 강하게 반영된다.
time_weighted_retriever = TimeWeightedVectorStoreRetriever(vectorstore=memo_vectorstore, decay_rate=0.01, k=1)

# 같은 주제(커피)를 다루지만 작성 시점이 다른 메모들. add_documents 의 current_time 으로 과거 시점을 지정한다.
time_weighted_retriever.add_documents(
    [Document(page_content="아메리카노는 진하고 깊은 맛의 클래식한 커피입니다. 카페인 함량도 높은 편입니다.")],
    current_time=now - datetime.timedelta(weeks=3),
)
time_weighted_retriever.add_documents(
    [Document(page_content="라떼는 우유가 들어가 부드럽고 마시기 편합니다.")],
    current_time=now - datetime.timedelta(weeks=1),
)
time_weighted_retriever.add_documents(
    [Document(page_content="오늘 발견한 카페의 콜드브루가 정말 맛있었어요.")],
    current_time=now,
)

if __name__ == "__main__":
    query = "아메리카노 추천해줘"  # 의미상으로는 3주 전 메모가 가장 유사하다

    # 참고용 : 시간 가중치 없이 순수 유사도로만 검색하면 어떤 순서로 나오는지
    print(f"Q: {query}\n")
    print("[순수 유사도 검색 결과]")
    for doc, score in memo_vectorstore.similarity_search_with_score(query, k=3):
        print(f"  ({score:.4f}) {doc.page_content}  [작성: {doc.metadata['created_at']}]")

    print("\n[TimeWeightedVectorStoreRetriever 검색 결과] (최근성이 반영됨)")
    for doc in time_weighted_retriever.invoke(query):
        print(f"  {doc.page_content}  [작성: {doc.metadata['created_at']}]")