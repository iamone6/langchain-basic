#
#   chatbot example
#

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate   # 프롬프트용 템플릿
from langchain_core.prompts import FewShotChatMessagePromptTemplate #few-shot 용 템플릿
from langchain_core.output_parsers import StrOutputParser   # String 으로 답변받기
from langchain_core.output_parsers import JsonOutputParser  # JSON 으로 답변받기

load_dotenv()

# temp 0.0(엄격) ~ 1.0(창의)
claude = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0.3)

# prompt 에 넣을 few-shot(모범답안) 설정
examples_for_few_shot = [
    {"input": "Python과 JAVA 중 어느 것이 더 나은가요?", "output": "몰라요"},
    {"input": "Python과 PHP 중 어느 것이 더 나은가요?", "output": "몰라요"},
]
few_shot = FewShotChatMessagePromptTemplate(
    examples = examples_for_few_shot,
    example_prompt=ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}"),
    ]),
)

prompt = ChatPromptTemplate.from_messages([
    # system message: 역할 부여
    ("system", "You are a CTO of Anthropic. You are a very good software engineer and you are very good at explaining things in simple terms. Answer in {language}."),
    # few-shot
    few_shot,
    # human message : 실제 프롬프트
    ("human", "다음 질문에 성실하게 답변해줘. {question}"),
])

#
#   1. chain.stream() 에서 실제 실행이 시작됨. chain.stream({dict}) 의 dict 가 prompt 의 {key} 에 매핑되면서 전달됨
#   2. prompt 는 system, human(넘겨받은 question으로 대체), language 추가되어 prompt 리스트를 리턴함
#   3. 2번의 리턴값을 받아 claude.stream() 을 실행 (chain = prompt | claude). 즉, prompt 와 claude를 연결한 chain을 사용하여 claude.stream() 에서 실제 API 호출이 발생함
#   4. claude.stream() 은 generator 를 리턴하며, for chunk in chain.stream() 에서 chunk 를 통해 streaming 결과를 받아올 수 있음
#   5. chain = prompt | cluade | StringOutputParser(), JsonOutputParser() 등이 올 수 있음. Pydantic 으로 받을수도 있음.
chain = prompt | claude

final_answer = []
# .stream() 대신 .invoke() 를 사용하면 streaming 없이 한 번에 결과를 받아올 수 있음.
for chunk in chain.stream({"language": "Korean", "question": "Korea and Japan which is better?"}):
    #
    #   기본은 chunk.content
    #   chain 에 StrOutputParser() 가 연결되어 있으면 chunk 만.
    #   chain 에 JsonOutputParser() 가 연결되어 있으면 .invoke()를 써라. result['summary'] 이런식으로 key를 통해 접근해야 함
    #   
    final_answer.append(chunk.content)
    print(chunk.content, end="", flush=True)
print("\nFinal Answer:", "".join(final_answer))
