#
#   Long Context Reorder : 검색된 문서(청크)들을 LLM에 넘기기 전에 순서를 재배치하는 document transformer 입니다.
#
#   왜 필요한가 ("Lost in the Middle" 현상, https://arxiv.org/abs/2307.03172) :
#       LLM은 프롬프트에 넣은 긴 컨텍스트 중 "맨 앞"과 "맨 뒤"에 있는 정보는 잘 참조하지만,
#       "중간"에 묻힌 정보는 상대적으로 놓치는 경향이 있습니다.
#       그런데 retriever 는 보통 유사도가 높은 순서대로 문서를 반환하므로,
#       Stuff 방식처럼 그 순서 그대로 컨텍스트에 이어붙이면 가장 관련도가 높은 2~3번째 문서가
#       컨텍스트의 중간 어딘가에 놓이게 되어 오히려 LLM이 놓치기 쉬운 위치에 배치되는 문제가 생깁니다.
#       LongContextReorder 는 문서를 관련도 내림차순으로 받아, 가장 관련도가 높은 문서들을
#       리스트의 "맨 앞"과 "맨 뒤"에 번갈아 배치하고 관련도가 낮은 문서들을 중간으로 밀어내어,
#       중요한 정보가 LLM이 잘 참조하는 위치에 오도록 재배열합니다.
#
#   언제 필요한가 :
#       - k 를 크게 잡아 검색된 문서 수가 많을 때(대략 5~10개 이상), 즉 컨텍스트 길이가 길어질 때 효과가 큽니다.
#       - Stuff 방식처럼 검색된 문서 전체를 한 번에 이어붙여 LLM에 전달하는 경우에 특히 유효합니다.
#       - k 가 작아 컨텍스트가 짧다면(ex: k=2~3) "중간"이라 할 위치 자체가 거의 없어 효과가 미미합니다.
#
#   아래는 rag_vectorstore.py 가 저장해둔 ./chroma_db 를 재사용하여, 재배치 전/후 순서를 비교하고
#   재배치된 컨텍스트로 Claude 의 답변까지 받아보는 예제입니다.
#
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
from langchain_chroma.vectorstores import Chroma
from langchain_community.document_transformers import LongContextReorder
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils import embeddings

load_dotenv()

claude = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)

# rag_vectorstore.py 가 저장해둔 vector store 를 그대로 재사용
chromaDB = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

reordering = LongContextReorder()

if __name__ == "__main__":
    query = "키다리 아저씨의 정체는 무엇인가?"

    # ./chroma_db 에는 (rag_vectorstore.py 를 id 지정 없이 여러 번 실행했던 이력 때문에) 같은 청크가
    # 중복 저장되어 있어, k 를 작게 주면 상위 결과가 전부 동일 청크로 채워진다.
    # 넉넉히 검색한 뒤 내용 기준으로 dedupe 하여 실제로 서로 다른 청크 8개를 확보한다.
    seen = set()
    docs = []
    for doc in chromaDB.similarity_search(query, k=300):
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            docs.append(doc)
        if len(docs) == 8:
            break

    reordered_docs = reordering.transform_documents(docs)

    print(f"Q: {query}\n")
    print("[재배치 전] 유사도 순위대로 나열 (가장 관련도 높은 문서가 2번째 자리)")
    for i, doc in enumerate(docs, 1):
        print(f"  {i}. {doc.page_content[:40]}")

    print("\n[재배치 후] 관련도 높은 문서가 맨 앞/맨 뒤로 이동")
    for i, doc in enumerate(reordered_docs, 1):
        print(f"  {i}. {doc.page_content[:40]}")

    # Stuff 방식 : 재배치된 순서 그대로 컨텍스트로 이어붙여 LLM에 전달
    prompt = ChatPromptTemplate.from_messages([
        ("system", "다음 컨텍스트를 참고하여 질문에 답변해줘.\n\n{context}"),
        ("human", "{question}"),
    ])
    chain = prompt | claude | StrOutputParser()
    context = "\n\n".join(doc.page_content for doc in reordered_docs)
    answer = chain.invoke({"context": context, "question": query})

    print(f"\nA: {answer}")
