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
#   2. ParentDocumentRetriever 예제
#       작은 child chunk 로 벡터 검색을 하되(검색 정확도↑), LLM 에게는 그 child chunk 가 속한
#       더 큰 parent chunk 를 컨텍스트로 넘겨줍니다(문맥 손실↓).
#       parent_splitter 를 지정하면 원본 문서를 먼저 parent chunk 로 나누고, 그 각 parent chunk 를
#       다시 child_splitter 로 잘게 나눕니다. parent_splitter 를 생략하면 원본 문서 전체가 parent 가 됩니다.
#       child chunk 의 metadata 에는 자동으로 parent_id(id_key) 가 채워지고,
#       vectorstore 에는 child chunk 만 임베딩되어 저장되며, docstore 에는 parent chunk 원문이 저장됩니다.
#
#       rag_vectorstore.py 의 ./chroma_db 는 이미 다른 방식(parent_id 없이)으로 청킹/저장되어 있어 재사용할 수 없으므로,
#       이 예제는 청킹부터 새로 하고, 저장소는 디스크에 남기지 않도록 모두 메모리(InMemoryStore, persist_directory 미지정 Chroma)를 사용합니다.
#
#       하나의 문서를 docstore(parent chunk용)에 저장하고, 그 문서를 child_splitter 로 잘게 나누어 vectorstore(child chunk용)에 저장한 뒤,
#       child chunk 를 검색하면 그 chunk 가 속한 parent chunk 를 docstore 에서 찾아 LLM 에게 컨텍스트로 전달합니다.
#

from dotenv import load_dotenv
from langchain_chroma.vectorstores import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.documents import Document
from langchain_core.stores import InMemoryStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils import embeddings

load_dotenv()

claude = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)

file = "./file/키다리아저씨.txt"
with open(file, "r", encoding="utf-8") as f:
    document = Document(page_content=f.read())

# parent : 문맥을 넉넉히 담을 만큼 큰 chunk / child : 벡터 검색 정확도를 위한 작은 chunk
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)

# child chunk 임베딩 전용 vector store. persist_directory 를 지정하지 않으면 메모리에만 존재한다.
child_vectorstore = Chroma(collection_name="parent_document_children", embedding_function=embeddings)

# parent chunk 원문을 담아두는 저장소 (메모리)
parent_docstore = InMemoryStore()

parent_document_retriever = ParentDocumentRetriever(
    vectorstore=child_vectorstore,
    docstore=parent_docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    search_kwargs={"k": 3},
)
parent_document_retriever.add_documents([document])

if __name__ == "__main__":
    query = "그 사람은 왜 정체를 숨겼을까?"  # 대명사로 애매하게 질문

    # child_vectorstore 에서 직접 검색 : 실제 유사도 매칭은 이 작은 child chunk 로 이루어진다.
    child_chunks = child_vectorstore.similarity_search(query, k=3)
    print(f"원본 질문: {query}\n")
    for i, child in enumerate(child_chunks, 1):
        parent_id = child.metadata["doc_id"]  # id_key 기본값. 이 child 가 속한 parent chunk 의 id
        parent = parent_docstore.mget([parent_id])[0]
        print(f"[검색된 child {i}] ({len(child.page_content)}자) {child.page_content}")
        print(f" -> 이 child 가 속한 parent ({len(parent.page_content)}자) 에 포함되어 있음:")
        print(f"    {'포함됨' if child.page_content in parent.page_content else '포함 안됨(경계에 걸침)'}\n")

    # parent_document_retriever 로 검색 : 위 child chunk 가 속한 parent chunk(중복 제거)를 대신 반환한다.
    parent_chunks = parent_document_retriever.invoke(query)
    print(f"parent chunk 크기 (글자 수): {[len(p.page_content) for p in parent_chunks]}")

    # Stuff 방식 : 검색된 parent chunk 전체를 컨텍스트로 한 번에 LLM에 전달하여 답변 생성
    prompt = ChatPromptTemplate.from_messages([
        ("system", "다음 컨텍스트를 참고하여 질문에 답변해줘.\n\n{context}"),
        ("human", "{question}"),
    ])
    chain = prompt | claude | StrOutputParser()
    context = "\n\n".join(chunk.page_content for chunk in parent_chunks)
    answer = chain.invoke({"context": context, "question": query})

    print(f"\nA: {answer}")