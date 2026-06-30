#
#   RAG(검색 증강 생성) documentLoaders 예제코드
#       - RAG 를 위해 다양한 포멧의 문서를 읽어오는 예제
#   DocumentLoaders -> TextSplitters(Chunking) ->  Embedding -> VectorStores -> RetrievalQA
#       WEB : WebBaseLoader(url) -> load(), UnstructuredURLLoader(urlList) -> load()
#       PDF : PyPDFLoader(filePath) -> load(), UnstructuredPDFLoader(filePath) -> load()
#       CSV: CsvLoader(filePath, csv_args={'delimiter': ',', 'quotechar': '"', 'fieldnames': ['column1', 'column2']} ) -> load(), UnstructuredCSVLoader(filePath) -> load()
#           csv 는 행마다 Document 객체 로 리턴된다.
#       DOCS : Docx2txtLoader(filePath) -> load(), UnstructuredWordDocumentLoader(filePath) -> load()
#       EXCEL : import openpyxl + langchain_core.document import Document 조합
#
import os
import warnings
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredExcelLoader, WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
#[deprecated] from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_core.documents import Document
import openpyxl



# document_loaders 가 deprecated 예고되었는데 아직 대체가 없어 warning 을 일단 숨김
warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()
# WebBaseKLoader의 User-Agent 설정
os.environ.setdefault("USER_AGENT", "langchain-basic/1.0")


#  WebBaseLoader 예제
wwwLoader = WebBaseLoader("https://n.news.naver.com/article/037/0000038423?cds=news_media_pc&type=editn")
data = wwwLoader.load()
wwwcContent: list = []
for line in data[0].page_content.splitlines():  # data[0].page_content 는 Document 객체.
    if len(line.strip()) > 0:
        wwwcContent.append(line.strip())
        # print(line.strip())
print("\n".join(wwwcContent))

# pdf loader 예제
pdfLoader = PyPDFLoader("./file/pdf-sample.pdf")
data = pdfLoader.load_and_split()   # page 단위로 나누어 load
pdfContent: list = []
for line in data[0].page_content.splitlines():
    if len(line.strip()) > 0:
        pdfContent.append(line.strip())
        # print(line.strip())
print("\n".join(pdfContent))

# docx loader 예제
docxLoader = Docx2txtLoader("./file/docx-sample.docx")
data = docxLoader.load()
docxContent: list = []
for line in data[0].page_content.splitlines():
    if len(line.strip()) > 0:
        docxContent.append(line.strip())
        # print(line.strip())
print("\n".join(docxContent))

# xlsx loader 예제
wb = openpyxl.load_workbook("./file/xlsx-sample.xlsx")  #WorkBook (excel 전체)
ws = wb.active  #WorkSheet (excel sheet 하나)
### 시트가 여러개 있는 경우:
# print(wb.sheetnames)  # ['Sheet1', 'Sheet2', '매출']
### 이름으로 특정 시트 접근
# ws = wb["매출"]

text = "\n".join(
    "\t".join(str(cell) for cell in row if cell is not None)
    for row in ws.iter_rows(values_only=True)
)
data = [Document(page_content=text)]
print(data[0].page_content)

