# Description: This script demonstrates how to use the RAG system with multiple PDF documents.
# The script loads PDF documents, processes them using OpenAI embeddings, and creates a FAISS vector store.
# It then sets up a retrieval-based QA system using the ChatOpenAI language model and the vector store.
# The user can ask questions about the documents, and the system will provide answers based on the retrieved information.
# The vector store can be saved for future use.
# The RAG model combines retrieval and generation to provide more accurate and informative answers to questions.

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from typing import List, Union

# Load environment variables
load_dotenv()

class RAGSystem:
    def __init__(self, pdf_paths: Union[str, List[str]]):
        """
        Initialize the RAG system with one or more PDF paths
        
        Args:
            pdf_paths: Either a single PDF path string or a list of PDF path strings
        """
        self.pdf_paths = [pdf_paths] if isinstance(pdf_paths, str) else pdf_paths
        self.load_documents()
        self.setup_retrieval_system()

    def load_documents(self):
        """Load and process the PDF documents"""
        self.documents = []
        for pdf_path in self.pdf_paths:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found at {pdf_path}")
            
            loader = PyPDFLoader(pdf_path)
            self.documents.extend(loader.load())

    def setup_retrieval_system(self):
        """Set up the retrieval system with embeddings and vector store"""
        embeddings = OpenAIEmbeddings()
        self.vector_store = FAISS.from_documents(self.documents, embeddings)
        retriever = self.vector_store.as_retriever()
        
        # Initialize language model and QA chain
        llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever
        )

    def ask_question(self, query: str) -> dict:
        """Ask a question about the documents"""
        response = self.qa_chain.invoke({"query": query})
        return response

    def save_vector_store(self, path="vector_store.index"):
        """Save the FAISS index for future use"""
        self.vector_store.save_local(path)
        print(f"Vector store saved to {path}")

def main():
    # Example usage with multiple PDFs
    pdf_paths = [
        "/Users/yigitaydede/Dropbox/Documents/Papers/FertilityMC/TurkeyFertility/Papers/background/5.pdf",
        "/Users/yigitaydede/Dropbox/Documents/Papers/FertilityMC/TurkeyFertility/Papers/background/6.pdf"
    ]
    rag = RAGSystem(pdf_paths)

    # Example question
    question = "Based on the documents, compare and contrast the fertility trends between urban and rural areas, and explain potential socioeconomic factors influencing these differences."
    response = rag.ask_question(question)
    print(f"Question: {question}")
    print(f"Response: {response}")

    # Save the vector store
    rag.save_vector_store()

if __name__ == "__main__":
    main()
