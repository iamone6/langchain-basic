#
#   embedding 으로 생성한 백터를 vector store(백터db) 에 저장하고, 
#   검색할 때는 query 를 embedding 하여 vector db 에서 유사한 벡터를 검색합니다.(Retrive)
#       vectordb : 여기서는 chroma 를 사용합니다. (poetry add chromadb)
#       유사도 측정과 검색을 위해 facbebook 의 faiss library(embedding model 과 독립적임) 를 사용합니다.
#   
#       faiss 로도 동일한 작업을 할 수 있습니다. (poetry add faiss-cpu)
#

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
from utils import count_tokens, embeddings
from enums.chunk_type import ChunkType
from rag_chunking import Chunking
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS


file: str = "./file/키다리아저씨.txt"
chunk_size: int = 1000
token_chunk_size: int = 300

chunking = Chunking(file=file, chunk_type=ChunkType.TOKEN, chunk_size=chunk_size, token_chunk_size=token_chunk_size)
# chunks = chunking.split_text()

# embed_documents() 는 list[str] 를 받아 list[list[float]] 를 반환 (for loop 불필요)
# vectors = embeddings.embed_documents(chunks)

# Chroma vector store 에 벡터 저장 (to ./chroma_db) : persist_directory 가 없으면 객체만 리턴된다. (vector store 저장은 persist_directory 지정 필요)
#   Chroma.from_texts() 는 list[str] 를 받아 list[list[float]] 를 생성하고, vector store 에 저장합니다.
#   Chroma.from_documents() 는 list[Document] 를 받아 list[list[float]] 를 생성하고, vector store 에 저장합니다.
vectorDB = Chroma.from_texts(texts=chunking.split_text(),   # list[str] 또는 list[Document] 를 받음(.from_ducuments())
                             embedding=embeddings, # 임베딩 모델 객체
                             persist_directory="./chroma_db")  # vector store 저장 경로

# 질문
query: str = "키다리 아저씨의 정체는 무엇인가?"
query2: str = "키다리 아저씨는 왜 몰래 도왔을까?"
query3: str = "키다리 아저씨의 도움을 받는 사람의 이름은?"

# .similarity_search(query=query, k=3) : 유사도만으로 검색
# .max_marginal_relevance_search(query=query, k=3, fetch_k=20, lambda_mult=0.5) : 다른문서를 더 참조하여 검색
#   fetch_k — MMR 계산을 위해 먼저 후보로 가져올 문서 수입니다. 이 중에서 최종적으로 k개를 선택합니다. fetch_k가 클수록 더 다양한 후보 중에서 고르게 됩니다.
#           - MMR(Maximal Marginal Relevance) 은 유사도가 높으면서도 서로 중복되지 않는 청크를 고르는 알고리즘입니다. 같은 내용이 반복되는 청크가 k개 모두 선택되는 걸 방지합니다.
#   lambda_mult — 유사도와 다양성 사이의 균형 조절값 (0.0 ~ 1.0)입니다.
answerDocs = vectorDB.similarity_search(query=query, k=3)  # k 는 검색할 유사도 상위 개수
print(f"chromaQ1: {query}")
print(f"chromaA1: {answerDocs[0].page_content}")  # 가장 유사한 문서 1개만 출력

#
#   저장된 chromaDB 객체를 로드하여 검색합니다.
#
chromaDB = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)  # vector store 로드
answerDocs2 = chromaDB.similarity_search(query=query2, k=3)  # k 는 검색할 유사도 상위 개수
# answerDocs2 = chromaDB.max_marginal_relevance_search(query=query2, k=3, fetch_k=20, lambda_mult=0.5)  # k 는 검색할 유사도 상위 개수
print(f"chromaQ2: {query2}")
print(f"chromaA2: {answerDocs2[0].page_content}")  # 가장 유사한 문서 1개만 출력
    
# score 가 작을수록 유사도가 높습니다. (0.0 ~ 1.0)
answerDocs3 = chromaDB.similarity_search_with_score(query=query3, k=3)
score = answerDocs3[0][1]  # 가장 유사한 문서 1개의 score
print(f"chromaQ3: {query3}")
print(f"chromaA3: {answerDocs3[0][0].page_content}")  # 가장 유사한 문서 1개만 출력
print(f"chromaScore3: {score}")

#
#   FAISS 를 사용하여 vector store 에 저장하고 검색합니다.
#

# .similarity_search(query=query, k=3) : 유사도만으로 검색
# .max_marginal_relevance_search(query=query, k=3, fetch_k=20, lambda_mult=0.5) : 다른문서를 더 참조하여 검색
#   fetch_k — MMR 계산을 위해 먼저 후보로 가져올 문서 수입니다. 이 중에서 최종적으로 k개를 선택합니다. fetch_k가 클수록 더 다양한 후보 중에서 고르게 됩니다.
#           - MMR(Maximal Marginal Relevance) 은 유사도가 높으면서도 서로 중복되지 않는 청크를 고르는 알고리즘입니다. 같은 내용이 반복되는 청크가 k개 모두 선택되는 걸 방지합니다.
#   lambda_mult — 유사도와 다양성 사이의 균형 조절값 (0.0 ~ 1.0)입니다.

faissDB = FAISS.from_texts(texts=chunking.split_text(), embedding=embeddings)  # vector store 생성
answerDocs4 = faissDB.max_marginal_relevance_search(query=query, k=3, fetch_k=20, lambda_mult=0.5)
print(f"faissQ1: {query}")
print(f"faissA1: {answerDocs4[0].page_content}")  # 가장 유사

faissDB.save_local("./faiss_db")  # vector store 저장
faissLocalDB = FAISS.load_local("./faiss_db", embeddings, allow_dangerous_deserialization=True)  # vector
answerDocs5 = faissLocalDB.max_marginal_relevance_search(query=query2, k=3, fetch_k=20, lambda_mult=0.5)
print(f"faissQ2: {query2}")
print(f"faissA2: {answerDocs5[0].page_content}")  # 가장 유사한 문서 1개만 출력

# max_marginal_relevance_search_with_score_by_vector() 는 query string 대신 query vector 를 받아서 검색합니다. (query string → query vector → vector store 검색)
query3_vector = embeddings.embed_query(query3)  # query 문자열 → 벡터
answerDocs6 = faissLocalDB.max_marginal_relevance_search_with_score_by_vector(embedding=query3_vector, k=3, fetch_k=20, lambda_mult=0.5)
print(f"faissQ3: {query3}")
print(f"faissA3: {answerDocs6[0][0].page_content}")  # 가장 유사한 문서 1개만 출력
print(f"faissScore3: {answerDocs6[0][1]}")  # 가장 유사한 문서 1개의 score

