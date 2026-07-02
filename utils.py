import tiktoken
from langchain_huggingface import HuggingFaceEmbeddings

# cl100k_base: GPT-4 / Claude 계열 근사 토크나이저 (오프라인, API 호출 없음)
# Claude의 실제 토크나이저와 완전히 같지는 않지만 실무에서 근사치로 사용
_tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    """ 토큰단위로 청킹을 하려면 문서의 토큰수를 측정할 수 있어야 합니다."""
    return len(_tokenizer.encode(text))

# 한국어 특화 sentence-transformers 모델 (공개 모델, 토큰 불필요)
embeddings = HuggingFaceEmbeddings(
    # 임베딩용 모델 정의
    model_name="jhgan/ko-sroberta-multitask",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
