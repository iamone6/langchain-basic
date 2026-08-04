#
#   sql agent 는 prompt 를 입력받아 sql을 실행하고 결과를 반환하는 역할을 수행합니다.
#   여기선 sqlite  - SQLDatabase.from_uri("sqlite:////data/chinook.db") - 의 예제를 사용합니다
#   /home/iamone/src/sqlite/Dockerfile , ./chinook.db 참고
#   동작 흐름:
#   1. DB 스키마 정보(테이블/컬럼)를 LLM에게 제공
#   2. 사용자가 자연어로 질문 (ex: "지난달 매출 top 5 고객은?")
#   3. LLM이 스키마를 보고 SQL 쿼리 생성
#   4. 생성된 쿼리를 실제 DB에서 실행
#   5. 실행 결과(row들)를 다시 LLM에 넣어 자연어로 요약/답변
#   6. 쿼리가 틀렸으면(에러) 에러 메시지를 보고 스스로 재작성해서 재시도 (agent 의 특징)
#
#   chinook-data 컨테이너(/home/iamone/src/sqlite/Dockerfile)는 볼륨만 제공하므로,
#   docker cp chinook-data:/data/chinook.db ./file/chinook.db 로 로컬에 꺼내와 사용합니다.
#
#   여기에 Q -> A 예제인 few shot을 질문 -> sql 인 dictionary 로 만들어 이를 vector db 에 저장하고,
#   검색된 chunk를 기반으로 sql을 생성하도록 하면, sql 생성 정확도를 높일 수 있습니다.
#   유사한 few-shot 이 없으면 기존 invoke 방식 그대로 사용합니다.
#   유사한 few-shot 이 있는경우
#
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase  # sqlite 파일을 감싸는 DB 래퍼 (db)
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit  # list_tables/schema/query 등 tool 생성 (toolkit)
from langchain_anthropic import ChatAnthropic  # Claude 모델 래퍼 (claude)
from langchain.agents import create_agent  # tool 을 사용하는 agent 생성 (agent)
from langchain_chroma import Chroma  # few-shot 질문을 저장/검색하는 vector db (fewShotDB)
from utils import embeddings  # few-shot 질문 임베딩에 사용하는 HuggingFace 임베딩 모델

load_dotenv()

# python 에는 sqlite3 모듈이 내장되어 있어 별도 설치 필요 없음
db = SQLDatabase.from_uri("sqlite:///./file/chinook.db")

# temperature=0 : SQL 생성은 정확성이 중요하므로 창의성 배제
claude = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)

# list_tables, get_schema, query_checker, run_query 등의 tool 을 제공
toolkit = SQLDatabaseToolkit(db=db, llm=claude)
agent = create_agent(model=claude, tools=toolkit.get_tools(), system_prompt="당신은 DBA 입니다.")


# 질문 -> SQL few-shot 예제. chinook.db 스키마에 자주 나올 법한 질문들을 미리 준비해둔다.
FEW_SHOT_EXAMPLES = [
    {
        "question": "가장 많이 팔린 트랙 top 5는?",
        "sql": 'SELECT t."Name", SUM(ii."Quantity") AS Sold FROM tracks t '
               'JOIN invoice_items ii ON t."TrackId" = ii."TrackId" '
               'GROUP BY t."TrackId" ORDER BY Sold DESC LIMIT 5;',
    },
    {
        "question": "가장 많이 지출한 고객 top 3는?",
        "sql": 'SELECT c."FirstName", c."LastName", SUM(i."Total") AS Spent FROM customers c '
               'JOIN invoices i ON c."CustomerId" = i."CustomerId" '
               'GROUP BY c."CustomerId" ORDER BY Spent DESC LIMIT 3;',
    },
    {
        "question": "장르별 트랙 개수는?",
        "sql": 'SELECT g."Name", COUNT(*) AS TrackCount FROM tracks t '
               'JOIN genres g ON t."GenreId" = g."GenreId" '
               'GROUP BY g."GenreId" ORDER BY TrackCount DESC;',
    },
    {
        "question": "직원 수는 총 몇 명인가?",
        "sql": 'SELECT COUNT(*) FROM employees;',
    },
    {
        "question": "가장 오래 근무한 직원은 누구인가?",
        "sql": 'SELECT "FirstName", "LastName", "HireDate" FROM employees ORDER BY "HireDate" ASC LIMIT 1;',
    },
    {
        "question": "국가별 고객 수는?",
        "sql": 'SELECT "Country", COUNT(*) AS CustomerCount FROM customers '
               'GROUP BY "Country" ORDER BY CustomerCount DESC;',
    },
]

# few-shot 전용 vector db (rag_vectorstore.py 의 chroma_db 와는 별개 저장소)
fewShotDB = Chroma.from_texts(
    texts=[example["question"] for example in FEW_SHOT_EXAMPLES],
    embedding=embeddings,
    metadatas=[{"sql": example["sql"]} for example in FEW_SHOT_EXAMPLES],
    persist_directory="./sql_fewshot_db",
    collection_metadata={"hnsw:space": "cosine"},  # score = 1 - cosine_similarity (0에 가까울수록 유사)
)

# score(거리)가 이 값 이하일 때만 few-shot 을 신뢰할 만큼 유사하다고 판단
FEW_SHOT_DISTANCE_THRESHOLD = 0.2


# vector db 에서 질문과 가장 유사한 few-shot 을 찾아, 임계값 이내일 때만 (question, sql) 로 반환
def search_fewshot(prompt: str) -> dict | None:
    results = fewShotDB.similarity_search_with_score(query=prompt, k=1)
    if not results:
        return None

    doc, score = results[0]
    if score > FEW_SHOT_DISTANCE_THRESHOLD:
        return None

    return {"question": doc.page_content, "sql": doc.metadata["sql"]}


# few-shot 이 있으면 prompt 뒤에 참고용 suffix 를 붙이고, 없으면 원본 prompt 그대로 사용.
# fewshot 매칭 여부도 함께 반환하여 ask() 에서 답변에 표시할 수 있게 한다.
def build_prompt(prompt: str) -> tuple[str, dict | None]:
    fewshot = search_fewshot(prompt)
    if fewshot is None:
        return prompt, None

    suffix = (
        f"\n\n(참고: 유사한 질문 '{fewshot['question']}'에는 다음 SQL 쿼리가 사용되었습니다. "
        f"이 쿼리를 참고하여 답변에 활용해줘.\n{fewshot['sql']})"
    )
    return prompt + suffix, fewshot


# agent.invoke() 는 한 번에 최종 답변과 중간에 실행된 tool_call 들을 모두 messages 로 반환하므로,
# 호출을 한 번만 하고 그 결과에서 답변과 SQL 쿼리를 함께 뽑아낼 수 있습니다.
def ask(prompt: str) -> tuple[str, str | None]:
    final_prompt, fewshot = build_prompt(prompt)
    result = agent.invoke({"messages": [("user", final_prompt)]})

    query = None
    for message in result["messages"]:
        for tool_call in getattr(message, "tool_calls", None) or []:
            if tool_call["name"] == "sql_db_query":
                query = tool_call["args"]["query"]

    answer = result["messages"][-1].content
    if fewshot is not None:
        answer += f"\n\n*(few-shot 예제 '{fewshot['question']}'가 참고되었습니다.)*"

    return answer, query


if __name__ == "__main__":
    for prompt in [
        "이 데이터베이스에는 어떤 테이블들이 있나요?",
        "가장 많이 팔린 트랙(Track) top 5는 무엇인가요?",
        "가장 많이 지출한 고객(Customer) 3명은 누구인가요?",
    ]:
        answer, sql_query = ask(prompt)
        print(f"Q: {prompt}")
        print(f"A: {answer}")
        print(f"SQL: {sql_query}\n")
