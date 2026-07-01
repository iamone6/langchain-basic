import tiktoken

# cl100k_base: GPT-4 / Claude 계열 근사 토크나이저 (오프라인, API 호출 없음)
# Claude의 실제 토크나이저와 완전히 같지는 않지만 실무에서 근사치로 사용
_tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(_tokenizer.encode(text))
