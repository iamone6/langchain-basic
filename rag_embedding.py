#
#   claude 는 임베딩용 api 를 제공하지 않고, openAI 는 유료버전만 사용가능해서 
#   Huggingface 의 무료 모델을 사용하여 임베딩을 생성합니다.
#   poetry add langchain-huggingface sentence-transformers
#       sentence-transformers — 실제 모델을 로드/실행하는 라이브러리
#       langchain-huggingface — LangChain용 래퍼
#

#
import os
from enums.chunk_type import ChunkType
from rag_chunking import Chunking
from langchain_huggingface import HuggingFaceEmbeddings

# cos 유사도 측정을 위해
from numpy import dot
from numpy.linalg import norm
import numpy as np

file: str = "./file/키다리아저씨.txt"
chunk_size: int = 1000
token_chunk_size: int = 300

# 한국어 특화 sentence-transformers 모델 (공개 모델, 토큰 불필요)
embeddings = HuggingFaceEmbeddings( model_name="jhgan/ko-sroberta-multitask",
                                    model_kwargs={"device": "cpu"},
                                    encode_kwargs={"normalize_embeddings": True})

chunking = Chunking(file=file, chunk_type=ChunkType.TOKEN, chunk_size=chunk_size, token_chunk_size=token_chunk_size)
chunks = chunking.split_text()

# embed_documents() 는 list[str] 를 받아 list[list[float]] 를 반환 (for loop 불필요)
vectors = embeddings.embed_documents(chunks)
print(f"청크 수: {len(vectors)}")
print(f"벡터 차원: {len(vectors[0])}")
print(f"첫 번째 벡터 앞 5개: {vectors[0][:5]}")

# cos similarity(코사인 유사도) 계산
def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    return dot(a, b) / (norm(a) * norm(b))

#   검색할 문장은 embed_query() 를 사용하고, 문서 임베딩은 embed_documents() 를 사용합니다.
embedded_query_vector = embeddings.embed_query("키다리 아저씨의 정체는 무엇인가?")
embedded_answer_vector = embeddings.embed_documents(["키다리 아저씨는 부자이며, 고아 소녀를 도와주는 인물입니다."])[0]
similarity = cos_sim(embedded_query_vector, embedded_answer_vector)
similarity2 = cos_sim(embedded_query_vector, vectors[100])
print(similarity)  # 0.8 이상이면 유사도가 높다고 판단
print(similarity2)  # 0.8 이상이면 유사도가 높다고 판단