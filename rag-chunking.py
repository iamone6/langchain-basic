#
#   rag - chunking 예제코드
#       chunking 은 CharacterTextSplitter 또는 RecursiveCharacterTextSplitter 를 이용하여 문서를 일정한 길이로 나누는 작업을 의미합니다.
#       CharacterTextSplitter 는 단순히 문서를 하나의 기준(ex:개행2개)으로만 쪼개며, 
#       RecursiveCharacterTextSplitter 는 여러 기준(ex:개행2개, 개행1개, 공백 등)을 순차적으로 재귀 적용하여 문서를 쪼갭니다.
#       둘 다 MAX_TOKEN_LENGTH 를 초과하는 문서는 쪼개지지 않으며, 
#       MAX_TOKEN_LENGTH 를 초과하는 문서는 RecursiveCharacterTextSplitter 로 정해진 토큰 길이 이하로 잘라야 합니다.
#       그래서 일반적으로 RecursiveCharacterTextSplitter 가 사용됩니다.
#
#       text를 쪼갤때는 splitter.split_text() 는 list[str] 를 리턴하며, 각 str 은 chunk 단위의 문서입니다.
#       pdf 와 같이 Document 객체를 쪼갤때는 spliotter.split_documents() 는 list[Document] 를 리턴하며, 각 Document 는 chunk 단위의 문서입니다.
#
#       일반 text 기반 문서가 아닌 코드,latex,html 등은 (Recuresive)CharacterTextSplitter 에 Language 를 추가로 import 및 지정하여 쪼개야 합니다. (ex: language=Language.PYTHON)
#
import os
from dotenv import load_dotenv
import tiktoken
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

load_dotenv()

# tiktoken 기준 length_function
# cl100k_base: GPT-4 / Claude 계열 근사 토크나이저 (오프라인, API 호출 없음)
# Claude의 실제 토크나이저와 완전히 같지는 않지만 실무에서 근사치로 사용
_tokenizer = tiktoken.get_encoding("cl100k_base")
def count_tokens(text: str) -> int:
    return len(_tokenizer.encode(text))

file = "./file/키다리아저씨.txt"
#  글자수 기준 청킹
chunk_size = 1000
# tiktoken 기준 청킹
token_chunk_size = 300  # 토큰 단위

# CharacterTextSplitter 사용객체 정의
splitter = CharacterTextSplitter(
    separator="\n\n",  # 문서를 쪼갤 기준
    chunk_size=chunk_size,  # 쪼갤 길이
    chunk_overlap=0,  # 쪼갤 때 겹치는 길이
    length_function=len,  # 길이를 측정할 함수
)

# RecursiveCharacterTextSplitter 사용객체 정의
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,  # 쪼갤 길이
    chunk_overlap=50,  # 쪼갤 때 겹치는 길이
    length_function=len,  #  길이를 측정할 함수
    separators=["\n\n", "\n", ".", ", ", " ", ""],  # 문서를 쪼갤 기준, 순차적으로 적용됨. "" 은 마지막으로 적용됨
)

# tiktoken 기준 RecursiveCharacterTextSplitter 사용객체 정의
token_splitter = RecursiveCharacterTextSplitter(
    chunk_size=token_chunk_size,
    chunk_overlap=20,
    length_function=count_tokens,  # 문자 수 대신 토큰 수 기준
    separators=["\n\n", "\n", ".", " ", ""],
)

with open(file, "r", encoding="utf-8") as f:
    text: str = f.read()
    print("문서 길이:", len(text), "문자")

    chunks = splitter.split_text(text)
    print("CharacterTextSplitter 청킹 개수:", len(chunks), "개")
    # for i, each_chunk in enumerate(chunks):
    #     if len(each_chunk) > chunk_size:
    #         raise ValueError(f"chunk {i+1} 길이({len(each_chunk)})가 chunk_size({chunk_size})를 초과했습니다. RecursiveCharacterTextSplitter 를 사용하세요.")
    
    recursive_chunks = recursive_splitter.split_text(text)
    print("recursive 문서 청킹 개수:", len(recursive_chunks), "개")
    for i, each_chunk in enumerate(recursive_chunks):
        if len(each_chunk) > chunk_size:
            print(f"chunk {i+1} 길이:", len(each_chunk), "문자")

    token_chunks = token_splitter.split_text(text)
    print(f"\ntiktoken 기준 청킹: {len(token_chunks)}개")
    print(f"첫 번째 청크 토큰 수: {count_tokens(token_chunks[0])}") 

