#
#   다양한 chunking 방식에 대한 설명과 예제 코드를 담은 파일입니다.
#
#   1. RecursiveCharacterTextSplitter : 계층적인 구분자(문단 -> 줄바꿈 -> 문장 -> 단어 순)를
#       순차적으로 적용하여 청크 크기를 chunk_size 이하로 최대한 유지하는 방식입니다.
#       특정 도메인에 종속되지 않아 범용적으로 가장 많이 사용됩니다. (rag_chunking.py 참고)
#
#   2. 코드 청킹 : RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, ...) 처럼
#       프로그래밍 언어를 지정하면, 그 언어의 문법 구조(클래스 정의, 함수 정의, 블록 등)를 아는
#       구분자 목록을 사용해 코드를 자릅니다. 함수/클래스 중간이 잘리는 걸 최대한 피할 수 있습니다.
#
#   3. MarkdownHeaderTextSplitter : 마크다운 문서를 `#`, `##`, `###` 같은 헤더 계층 구조를 기준으로
#       나눕니다. 각 청크의 metadata 에 그 청크가 속한 상위 헤더 경로(예: {"h1": "...", "h2": "..."})가
#       자동으로 채워져서, 문서의 구조 정보를 chunk 에 함께 담을 수 있습니다.
#
#   4. Semantic Chunking(SemanticChunker) : 글자/토큰 수 같은 고정 길이 기준이 아니라,
#       문장 단위로 임베딩한 뒤 "인접 문장 간 의미 차이가 크게 벌어지는 지점"을 경계로 나눕니다.
#       의미가 이어지는 문장들은 한 청크에, 주제가 바뀌는 지점에서 청크가 나뉘어 문맥 일관성이 높습니다.
#       (langchain_experimental 패키지 필요 poetry add langchain-experimental, 임베딩 모델을 써서 청킹 시점에도 비용이 듦)
#
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter, Language
from langchain_experimental.text_splitter import SemanticChunker
from utils import embeddings

if __name__ == "__main__":
    file = "./file/키다리아저씨.txt"
    with open(file, "r", encoding="utf-8") as f:
        text = f.read()

    # 1. RecursiveCharacterTextSplitter
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", ", ", " ", ""],
    )
    recursive_chunks = recursive_splitter.split_text(text)
    print(f"[1. RecursiveCharacterTextSplitter] {len(recursive_chunks)}개 청크")
    print(f"  첫 번째 청크: {recursive_chunks[0][:80]}\n")

    # 2. 코드 청킹 : Language.PYTHON 을 지정하면 "class", "def" 등 파이썬 문법 단위를 우선 구분자로 사용
    python_code = '''
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        raise NotImplementedError


class Dog(Animal):
    def speak(self):
        return f"{self.name}: 멍멍!"


def main():
    dog = Dog("바둑이")
    print(dog.speak())
'''
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=100,
        chunk_overlap=0,
    )
    code_chunks = python_splitter.split_text(python_code)
    print(f"[2. 코드 청킹 (Language.PYTHON)] {len(code_chunks)}개 청크")
    for i, chunk in enumerate(code_chunks, 1):
        print(f"  청크{i}: {chunk!r}")
    print()

    # 3. MarkdownHeaderTextSplitter : 헤더 레벨과 그 레벨을 metadata 에 저장할 때 쓸 key 이름을 매핑
    markdown_text = """# 키다리 아저씨
## 등장인물
### 주디
고아원 출신의 주인공.
### 키다리 아저씨
정체를 숨긴 후원자.
## 줄거리
편지 형식으로 전개되는 성장 소설.
"""
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
    )
    markdown_chunks = markdown_splitter.split_text(markdown_text)
    print(f"[3. MarkdownHeaderTextSplitter] {len(markdown_chunks)}개 청크")
    for chunk in markdown_chunks:
        print(f"  metadata={chunk.metadata} content={chunk.page_content!r}")
    print()

    # 4. Semantic Chunking : 문장을 임베딩하여 의미가 크게 바뀌는 지점을 경계로 나눔
    semantic_splitter = SemanticChunker(embeddings)
    semantic_chunks = semantic_splitter.split_text(text[:3000])  # 임베딩 비용 때문에 앞부분만 사용
    print(f"[4. SemanticChunker] {len(semantic_chunks)}개 청크")
    print(f"  첫 번째 청크: {semantic_chunks[0][:80]}")
