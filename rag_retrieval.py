#
#   rag_vectorstore.py 에서 저장된 vector store 를 로드하여 검색하는 예제코드입니다.
#
#   컨텍스트 Chain 의 4가지 종류:
#   Stuff : [{'Q': query, 'Context': 검색된 Chunks}] 를 넣어 LLM 에 던진다
#   Map Reduce : 각 Chunk들을 병렬로 LLM에 요약을 요청 (MAP) -> 요약된 LLM 결과들을 모아 다시 LLM에 넣어 최종 요약을 요청받고(REDUCE),
#       [{'Q': query, 'Context': 최종요약Chunk}] 로 LLM에 던진다
#   Refine(고품질) : 각 Chunk 를 순차적으로 [{'Q': query, 'intermediate answer' : Chunk[i-1], 'Context': Chunk[i]}] 로 LLM에 answer 를 요청하고,
#       그 답변을 intermediate answer 에 넣어 마지막까지 루프를 돌려 나온 답변이 최종
#   Map Rerank : 각 Chunk들을 병렬로 LLM에 score 와 함께 answer 를 요청한 뒤 나온 결과중 가장 높은 score 를 가진 answer 를 최종 answer 로 선택한다.
#
import os
from dotenv import load_dotenv
from langchain_chroma.vectorstores import Chroma
from langchain_anthropic import ChatAnthropic   # 앤스로픽 모델을 사용하기 위한 래퍼
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from utils import embeddings  # huggingface 의 embedding model 객체
from enums.chain_type import ChainType

load_dotenv()

# vector store 로드
chromaDB = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Claude 모델을 사용하기 위한 래퍼 객체 정의
claude = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0.3)  # temp(0~1) 가 0에 가까울수록 일관적으로 답변


class Retrieval:
    def __init__(self, vectorDB: Chroma, llm: ChatAnthropic, chain_type: ChainType, k: int = 3):
        self.retriever = vectorDB.as_retriever(search_kwargs={"k": k})
        self.llm = llm
        self.chain_type = chain_type

    def answer(self, query: str) -> str:
        docs = self.retriever.invoke(query)
        if self.chain_type == ChainType.STUFF:
            return self._stuff(query, docs)
        elif self.chain_type == ChainType.MAP_REDUCE:
            return self._map_reduce(query, docs)
        elif self.chain_type == ChainType.REFINE:
            return self._refine(query, docs)
        elif self.chain_type == ChainType.MAP_RERANK:
            return self._map_rerank(query, docs)

    # Stuff : 검색된 Chunk 전체를 컨텍스트로 한 번에 LLM에 전달
    def _stuff(self, query: str, docs: list) -> str:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "다음 컨텍스트를 참고하여 질문에 답변해줘.\n\n{context}"),
            ("human", "{question}"),
        ])
        chain = prompt | self.llm | StrOutputParser()
        context = "\n\n".join(doc.page_content for doc in docs)
        return chain.invoke({"context": context, "question": query})

    # Map Reduce : Chunk 마다 관련 정보를 병렬로 추출(MAP)한 뒤, 모아서 최종 답변을 생성(REDUCE)
    def _map_reduce(self, query: str, docs: list) -> str:
        map_prompt = ChatPromptTemplate.from_messages([
            ("human", "다음 컨텍스트에서 '{question}'과 관련된 정보만 추출해줘.\n\n컨텍스트: {context}"),
        ])
        map_chain = map_prompt | self.llm | StrOutputParser()
        summaries = map_chain.batch([{"question": query, "context": doc.page_content} for doc in docs])

        reduce_prompt = ChatPromptTemplate.from_messages([
            ("system", "다음 요약들을 참고하여 질문에 답변해줘.\n\n{context}"),
            ("human", "{question}"),
        ])
        reduce_chain = reduce_prompt | self.llm | StrOutputParser()
        return reduce_chain.invoke({"context": "\n\n".join(summaries), "question": query})

    # Refine : Chunk 를 순차적으로 돌며 이전 답변(intermediate answer)을 새 컨텍스트로 개선
    def _refine(self, query: str, docs: list) -> str:
        refine_prompt = ChatPromptTemplate.from_messages([
            ("human", "질문: {question}\n기존 답변: {existing_answer}\n새 컨텍스트: {context}\n\n새 컨텍스트를 반영하여 답변을 개선해줘."),
        ])
        refine_chain = refine_prompt | self.llm | StrOutputParser()

        answer = "(아직 답변 없음)"
        for doc in docs:
            answer = refine_chain.invoke({"question": query, "existing_answer": answer, "context": doc.page_content})
        return answer

    # Map Rerank : Chunk 마다 병렬로 (answer, score) 를 생성한 뒤, 가장 score 가 높은 answer 를 채택
    def _map_rerank(self, query: str, docs: list) -> str:
        rerank_prompt = ChatPromptTemplate.from_messages([
            ("human",
             "다음 컨텍스트로 질문에 답변하고, 답변의 확신도를 0~100 점수로 평가하여 "
             '{{"answer": "...", "score": 0}} 형식의 JSON으로만 반환해줘.\n\n'
             "질문: {question}\n컨텍스트: {context}"),
        ])
        rerank_chain = rerank_prompt | self.llm | JsonOutputParser()
        results = rerank_chain.batch([{"question": query, "context": doc.page_content} for doc in docs])
        best = max(results, key=lambda r: r["score"])
        return best["answer"]


# 질문
query: str = "키다리 아저씨의 정체는 무엇인가?"
query2: str = "키다리 아저씨는 왜 몰래 도왔을까?"
query3: str = "키다리 아저씨의 도움을 받는 사람의 이름은?"

if __name__ == "__main__":
    for q in [query, query2, query3]:
        for chain_type in ChainType:
            retrieval = Retrieval(vectorDB=chromaDB, llm=claude, chain_type=chain_type)
            print(f"[{chain_type.value}] Q: {q}")
            print(f"[{chain_type.value}] A: {retrieval.answer(q)}\n")
