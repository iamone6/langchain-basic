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
#       pdf 와 같이 Document 객체를 쪼갤때는 splitter.split_documents() 는 list[Document] 를 리턴하며, 각 Document 는 chunk 단위의 문서입니다.
#
#       일반 text 기반 문서가 아닌 코드,latex,html 등은 (Recuresive)CharacterTextSplitter 에 Language 를 추가로 import 및 지정하여 쪼개야 합니다. (ex: language=Language.PYTHON)
#
import os
from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from enums.chunk_type import ChunkType
from tiktoken_utils import count_tokens

load_dotenv()

file = "./file/키다리아저씨.txt"
#  글자수 기준 청킹
chunk_size = 1000
# tiktoken 기준 청킹
token_chunk_size = 300  # 토큰 단위

class Chunking:
    def __init__(self, file: str, chunk_type: ChunkType, chunk_size: int = 1000, token_chunk_size: int = 300):
        self.file = file
        self.chunk_type = chunk_type
        self.chunk_size = chunk_size
        self.token_chunk_size = token_chunk_size

    def split_text(self):
        with open(self.file, "r", encoding="utf-8") as f:
            text: str = f.read()
            print("문서 길이:", len(text), "문자")

            if self.chunk_type == ChunkType.NORMAL:
                # CharacterTextSplitter 사용객체 정의
                splitter = CharacterTextSplitter(
                    separator="\n\n",  # 문서를 쪼갤 기준
                    chunk_size=self.chunk_size,  # 쪼갤 길이
                    chunk_overlap=0,  # 쪼갤 때 겹치는 길이
                    length_function=len,  # 길이를 측정할 함수
                )
            elif self.chunk_type == ChunkType.RECURSIVE:
                # RecursiveCharacterTextSplitter 사용객체 정의
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,  # 쪼갤 길이
                    chunk_overlap=50,  # 쪼갤 때 겹치는 길이
                    length_function=len,  #  길이를 측정할 함수
                    separators=["\n\n", "\n", ".", ", ", " ", ""],  # 문서를 쪼갤 기준, 순차적으로 적용됨. "" 은 마지막으로 적용됨
                )
            elif self.chunk_type == ChunkType.TOKEN:
                # RecursiveCharacterTextSplitter 사용객체 정의
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,  # 쪼갤 길이
                    chunk_overlap=50,  # 쪼갤 때 겹치는 길이
                    length_function=count_tokens,  #  길이를 측정할 함수
                    separators=["\n\n", "\n", ".", " ", ""],
                )

            chunks = splitter.split_text(text)
            print("CharacterTextSplitter 청킹 개수:", len(chunks), "개")
            print(f"첫 번째 청크 토큰 수: {count_tokens(chunks[0])}")
            return chunks


chunking = Chunking(file=file, chunk_type=ChunkType.TOKEN, chunk_size=chunk_size, token_chunk_size=token_chunk_size)
chunking.split_text()