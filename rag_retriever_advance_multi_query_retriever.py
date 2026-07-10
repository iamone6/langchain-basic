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
#   1. MultiQueryRetriever 예제
#       질문 1개를 LLM 에게 넘겨 다른 표현의 질문 여러 개(기본 3개)로 변형시킨 뒤,
#       각 질문으로 vector db 를 검색하고, 중복을 제거한 문서 합집합을 반환합니다.
#       사용자가 애매하게 질문해도 표현이 다른 질문들로 재검색하기 때문에 recall 이 올라갑니다.
#       (langchain 1.x 에서 MultiQueryRetriever 는 langchain_classic 패키지로 이동됨)
#
from dotenv import load_dotenv
from langchain_chroma.vectorstores import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils import embeddings

load_dotenv()

# rag_vectorstore.py 에서 저장해둔 vector store 로드
chromaDB = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
claude = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)

base_retriever = chromaDB.as_retriever(search_kwargs={"k": 3})

# base_retriever 를 감싸서, 검색 전에 질문을 여러 버전으로 확장하는 retriever
multi_query_retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=claude)

if __name__ == "__main__":
    query = "그 사람은 왜 정체를 숨겼을까?"  # 대명사로 애매하게 질문

    # invoke() 는 내부적으로 generate_queries() -> retrieve_documents() -> unique_union() 순으로 동작하는데,
    # 생성된 질문 목록을 함께 보여주기 위해 콜백 없이 각 단계를 직접 호출한다. (invoke() 와 동일한 결과, LLM 호출은 1회)
    run_manager = CallbackManagerForRetrieverRun.get_noop_manager()
    generated_queries = multi_query_retriever.generate_queries(query, run_manager)
    docs = multi_query_retriever.unique_union(
        multi_query_retriever.retrieve_documents(generated_queries, run_manager)
    )

    print(f"원본 질문: {query}")
    print("생성된 질문:")
    for i, generated_query in enumerate(generated_queries, 1):
        print(f"  {i}. {generated_query}")

    print(f"검색된 문서 수 (중복 제거): {len(docs)}")

    # Stuff 방식 : 검색된 청크 전체를 컨텍스트로 한 번에 LLM에 전달하여 답변 생성
    prompt = ChatPromptTemplate.from_messages([
        ("system", "다음 컨텍스트를 참고하여 질문에 답변해줘.\n\n{context}"),
        ("human", "{question}"),
    ])
    chain = prompt | claude | StrOutputParser()
    context = "\n\n".join(doc.page_content for doc in docs)
    answer = chain.invoke({"context": context, "question": query})

    print(f"\nA: {answer}")
