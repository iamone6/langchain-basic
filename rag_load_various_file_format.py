#
#   RAG 를 위해 ./file 의 다양한 포맷(docx, pdf, xlsx, txt)과 web url 로부터 문서를 가져오는 예제입니다.
#   DocumentLoaders -> TextSplitters(Chunking) -> Embedding -> VectorStores -> Retrieval 파이프라인의 첫 단계입니다.
#   각 포맷을 최종적으로 langchain_core.documents.Document 객체(list)로 통일해서 반환합니다.
#
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)  # document_loaders deprecated 예고 숨김

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("USER_AGENT", "langchain-basic/1.0")  # WebBaseLoader 의 User-Agent 설정 (import 전에 지정해야 경고가 안 뜸)

from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, Docx2txtLoader
import openpyxl

if __name__ == "__main__":
    # TXT : 별도 Loader 클래스가 필요 없습니다. 파일 전체가 순수 텍스트이므로
    #       파이썬 내장 open() 으로 읽어서 Document 로 감싸주기만 하면 됩니다.
    txt_path = "./file/키다리아저씨.txt"
    with open(txt_path, "r", encoding="utf-8") as f:
        txt_documents = [Document(page_content=f.read(), metadata={"source": txt_path})]
    print(f"[TXT] {len(txt_documents)}개 Document / 앞부분: {txt_documents[0].page_content[:50]}")

    # DOCX : Docx2txtLoader(docx2txt 패키지 필요) 를 사용합니다. load() 는 문서 전체를
    #       하나의 Document 로 반환합니다 (워드는 PDF 처럼 뚜렷한 "페이지" 개념이 없어 페이지 단위 분리를 지원하지 않음).
    docx_path = "./file/docx-sample.docx"
    docx_documents = Docx2txtLoader(docx_path).load()
    print(f"[DOCX] {len(docx_documents)}개 Document / 앞부분: {docx_documents[0].page_content[:50]}")

    # PDF : PyPDFLoader(pypdf 패키지 필요) 를 사용합니다. load() 는 전체를 하나로 합쳐 반환하고,
    #       load_and_split() 은 PDF 의 페이지 구분을 살려 페이지 단위로 나눈 Document list 를 반환합니다.
    pdf_path = "./file/pdf-sample.pdf"
    pdf_documents = PyPDFLoader(pdf_path).load_and_split()
    print(f"[PDF] {len(pdf_documents)}개 Document(페이지 단위) / 앞부분: {pdf_documents[0].page_content[:50]}")

    # XLSX : 전용 Loader 가 없습니다 (UnstructuredExcelLoader 는 unstructured 패키지가 필요한데
    #       Python 3.14 를 지원하지 않음). openpyxl 로 직접 시트를 읽어 탭/개행으로 이어붙인 뒤
    #       Document 로 감싸는 방식으로 대체합니다.
    xlsx_path = "./file/xlsx-sample.xlsx"
    wb = openpyxl.load_workbook(xlsx_path)  # WorkBook : 엑셀 파일 전체
    ws = wb.active  # WorkSheet : 그 중 시트 하나 (기본은 활성 시트)
    xlsx_text = "\n".join(
        "\t".join(str(cell) for cell in row if cell is not None)
        for row in ws.iter_rows(values_only=True)
    )
    xlsx_documents = [Document(page_content=xlsx_text, metadata={"source": xlsx_path})]
    print(f"[XLSX] {len(xlsx_documents)}개 Document / 앞부분: {xlsx_documents[0].page_content[:50]}")

    # WEB : WebBaseLoader(beautifulsoup4 패키지 필요) 를 사용합니다. url 을 받아 HTML 을 파싱해
    #       본문 텍스트를 하나의 Document 로 반환합니다. User-Agent 미설정 시 경고가 발생하므로
    #       모듈 로딩 전에 os.environ 으로 미리 지정해둡니다.
    web_url = "https://n.news.naver.com/article/037/0000038423?cds=news_media_pc&type=editn"
    web_documents = WebBaseLoader(web_url).load()
    print(f"[WEB] {len(web_documents)}개 Document / 앞부분: {web_documents[0].page_content[:50]}")
