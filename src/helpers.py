from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

from dotenv import load_dotenv

load_dotenv()
AZURE_DEPLOYMENT_EMB=os.getenv("AZURE_DEPLOYMENT_EMB")
AZURE_OPENAI_ENDPOINT=os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY=os.getenv("AZURE_OPENAI_API_KEY")
class DocumentProcessor:
    def __init__(self):
        self.embedding = AzureOpenAIEmbeddings(
            chunk_size=16,
            azure_deployment=AZURE_DEPLOYMENT_EMB,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY
        )
    def load_split_pdf(self, file_path):
        pdf_loader = PyPDFLoader(file_path)
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(pdf_loader.load())
        return docs
    def create_and_save_db(self, docs):
        db = FAISS.from_documents(docs, self.embedding)
        db.save_local("faiss_index")
        return db
    def load_db(self):
        db= FAISS.load_local("faiss_index", self.embedding, allow_dangerous_deserialization=True)
        return db
    

# processer=DocumentProcessor()
# # docs=processer.load_split_pdf("Hamburg2.pdf")
# db=processer.load_db()
# query = "What did the president say about Ketanji Brown Jackson"


# docs = db.similarity_search(query)
# print(docs[0].page_content)

    

# def load_pdf(uploaded_file):
#     """Load the PDF document and extract the text."""
#     pdf_loader = PyPDFLoader(uploaded_file)
#     return pdf_loader.load()
# documents = load_pdf("Hamburg2.pdf")
# #documents = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# docs = text_splitter.split_documents(documents)
# embedding = AzureOpenAIEmbeddings(
#             chunk_size=16,
#             azure_deployment=AZURE_DEPLOYMENT_EMB,
#             azure_endpoint=AZURE_OPENAI_ENDPOINT,
#             api_key=AZURE_OPENAI_API_KEY
#         )
# db = FAISS.from_documents(docs, embedding)
# # print(db.index.ntotal)
# # query = "What did the president say about Ketanji Brown Jackson"
# # # docs = db.similarity_search(query)
# # # print(docs[0].page_content)

# db.save_local("faiss_index")




# new_db = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)

# docs = new_db.similarity_search(query)
# print(docs[0].page_content)