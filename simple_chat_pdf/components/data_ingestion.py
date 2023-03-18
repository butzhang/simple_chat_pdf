from typing import List, Optional
from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain.vectorstores import Pinecone
import pinecone
from langchain.embeddings.base import Embeddings

class DataIngestion:
    def __init__(self, pinecone_api_key: str, pinecone_environment: str, openai_api_key: str):
        self.embeddings: OpenAIEmbeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
        self.index_name = "langchain-demo"

    def read_pdf(self, pdf_file: str, text_splitter: Optional[TextSplitter] = None) -> List[Document]:
        """
        Read the content of a PDF file using the langchain library and split the text using the provided text_splitter.

        :param pdf_file: The PDF file to be read.
        :param text_splitter: Optional TextSplitter to split the text.
        :return: The pages extracted from the PDF.
        """
        loader = PyPDFLoader(pdf_file)
        pages = loader.load_and_split(text_splitter)
        return pages

    # def transform_documents_to_embeddings(self, documents: List[Document]):
    #     """
    #     Transform the documents obtained from read_pdf into embeddings.

    #     :param documents: The documents obtained from read_pdf.
    #     :return: A list of embeddings.
    #     """
    #     texts = [doc.page_content for doc in documents]  # Use page_content instead of text
    #     embeddings = self.embeddings.embed_documents(texts)
    #     return embeddings


    def send_embeddings_to_vector_store(self, documents: List[Document], embeddings: Embeddings):
        """
        Send the embeddings to a Pinecone vector store.

        :param documents: The documents to be stored in the vector store.
        :param embeddings: The embeddings to be sent to a vector store.
        :return: A status indicating success or failure.
        """
        docsearch = Pinecone.from_documents(documents, embeddings, index_name=self.index_name)
        # Code to send the embeddings to a Pinecone vector store
        return "success"  # Return a status indicating success or failure

    def parse_pdf(self, pdf_file: str) -> str:
        """
        Read a PDF file, transform its content into embeddings, and send the embeddings to a vector store.

        :param pdf_file: The PDF file to be parsed.
        :return: A status indicating success or failure.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20,
            length_function=len,
        )
        pages: List[Document] = self.read_pdf(pdf_file, text_splitter)
        status = self.send_embeddings_to_vector_store(pages, self.embeddings)
        return status
