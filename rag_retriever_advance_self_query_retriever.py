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
#   3. SelfQueryRetriever 예제
#       질문을 LLM 이 분석하여 "의미 검색용 텍스트"와 "metadata 필터 조건(구조화 쿼리)"으로 분리한 뒤,
#       그 필터를 만족하는 문서들 중에서만 vector similarity 검색을 수행합니다.
#       (ex: "평점 4.5 이상인 로맨스 소설" -> 검색어="로맨스" + 필터(genre=로맨스, rating>=4.5))
#       키다리아저씨.txt 는 metadata 가 없는 순수 텍스트라 필터링할 대상이 없어 이 기법의 장점을 보여줄 수 없으므로,
#       genre/year/rating 등 metadata 를 임의로 채운 소설 소개 Document 를 직접 정의해서 사용합니다.
#       metadata_field_info(AttributeInfo) 로 필터 가능한 필드와 설명을 LLM 에게 알려줘야 하며,
#       구조화 쿼리 파싱에 lark 패키지가 필요합니다 (poetry add lark).
#
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
from langchain_chroma.vectorstores import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_community.query_constructors.chroma import ChromaTranslator
from langchain_core.documents import Document
from utils import embeddings

load_dotenv()

claude = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)

# metadata(genre/year/rating)를 임의로 채운 소설 소개 Document
novels = [
    Document(
        page_content="고아 소녀 주디가 익명의 후원자 '키다리 아저씨'의 도움으로 대학에 진학하며 성장해가는 이야기. 편지 형식으로 쓰여졌다.",
        metadata={"title": "키다리 아저씨", "author": "진 웹스터", "genre": "성장소설", "year": 1912, "rating": 4.5},
    ),
    Document(
        page_content="엄격한 신분과 편견 속에서도 서로에게 이끌리는 엘리자베스와 다아시의 이야기를 그린 사교계 로맨스.",
        metadata={"title": "오만과 편견", "author": "제인 오스틴", "genre": "로맨스", "year": 1813, "rating": 4.7},
    ),
    Document(
        page_content="죽은 조직으로 생명을 창조하려는 과학자와 그가 만들어낸 존재의 비극을 다룬 이야기.",
        metadata={"title": "프랑켄슈타인", "author": "메리 셸리", "genre": "SF", "year": 1818, "rating": 4.3},
    ),
    Document(
        page_content="토끼굴에 빠진 소녀가 기묘한 인물들이 사는 이상한 나라를 모험하는 이야기.",
        metadata={"title": "이상한 나라의 앨리스", "author": "루이스 캐럴", "genre": "판타지", "year": 1865, "rating": 4.2},
    ),
    Document(
        page_content="늙은 어부가 거대한 청새치와 사흘간 사투를 벌이며 인간의 존엄을 보여주는 이야기.",
        metadata={"title": "노인과 바다", "author": "어니스트 헤밍웨이", "genre": "문학", "year": 1952, "rating": 4.4},
    ),
    Document(
        page_content="트란실바니아의 흡혈 백작이 런던으로 건너와 벌이는 공포와 사투를 그린 이야기.",
        metadata={"title": "드라큘라", "author": "브램 스토커", "genre": "공포", "year": 1897, "rating": 4.1},
    ),
    Document(
        page_content="남북전쟁 시기, 네 자매가 가난 속에서도 서로 의지하며 성장해가는 이야기.",
        metadata={"title": "작은 아씨들", "author": "루이자 메이 올컷", "genre": "성장소설", "year": 1868, "rating": 4.6},
    ),
    Document(
        page_content="한 사람 안에 선과 악이 분리되어 존재할 수 있는지를 실험한 의사의 비극적 이야기.",
        metadata={"title": "지킬 박사와 하이드", "author": "로버트 루이스 스티븐슨", "genre": "공포", "year": 1886, "rating": 4.0},
    ),
]

# persist_directory 를 지정하지 않으면 메모리에만 존재한다. 이 한줄로 임베딩 및 저장이 진행된다.
novel_vectorstore = Chroma.from_documents(documents=novels, embedding=embeddings, collection_name="novels")

# 필터링 가능한 metadata 필드와 설명 (LLM 이 구조화 쿼리를 만들 때 참고)
metadata_field_info = [
    AttributeInfo(name="genre", description="소설의 장르. 가능한 값: 성장소설, 로맨스, SF, 판타지, 문학, 공포", type="string"),
    AttributeInfo(name="year", description="소설이 출간된 연도", type="integer"),
    AttributeInfo(name="rating", description="5점 만점 기준 평점", type="float"),
]
document_contents = "소설의 줄거리 요약"

# structured_query_translator 를 생략하면 SelfQueryRetriever 가 vectorstore 종류에 맞는 translator 를
# 자동으로 고르려고 모든 벡터스토어용 translator 를 한꺼번에 import 하는데, 이 langchain-community 버전에서는
# 그 중 일부(Databricks) import 가 깨져 있어 오류가 난다. Chroma 용 translator 를 직접 지정해 우회한다.
self_query_retriever = SelfQueryRetriever.from_llm(
    llm=claude,
    vectorstore=novel_vectorstore,
    document_contents=document_contents,
    metadata_field_info=metadata_field_info,
    structured_query_translator=ChromaTranslator(),
)

if __name__ == "__main__":
    for query in [
        "1900년 이전에 나온 소설 중에서 평점이 4.5 이상인 것은?",
        "공포 장르 소설을 추천해줘",
    ]:
        print(f"Q: {query}")
        for doc in self_query_retriever.invoke(query):
            print(f"  - {doc.metadata['title']} ({doc.metadata['year']}, {doc.metadata['genre']}, 평점 {doc.metadata['rating']})")
        print()