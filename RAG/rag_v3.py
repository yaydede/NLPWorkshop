# This is to show that this is indeed a RAG (Retrieval-Augmented Generation) implementation, not just retrieval. Let me explain why:
# This part handles the retrieval
    # embeddings = OpenAIEmbeddings()
    # vector_store = FAISS.from_documents(documents, embeddings)
    # retriever = vector_store.as_retriever()
# This part handles the generation
    # llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)
    # qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
# If it were just retrieval, we wouldn't need the generation part. The generation part is what makes it a RAG model.
# The generation part is responsible for generating answers based on the retrieved information. This is the key feature of the RAG model.
# The retrieval part retrieves relevant information from the documents, and the generation part generates answers based on that information.
# The RAG model combines both retrieval and generation to provide more accurate and informative answers to questions.

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

class RAGSystem:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.load_documents()
        self.setup_retrieval_system()

    def load_documents(self):
        loader = PyPDFLoader(self.pdf_path)
        self.documents = loader.load()

    def setup_retrieval_system(self):
        embeddings = OpenAIEmbeddings()
        self.vector_store = FAISS.from_documents(self.documents, embeddings)
        self.retriever = self.vector_store.as_retriever()
        
        llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True  # This will show us the retrieved context
        )

    def ask_question(self, query):
        """Ask a question and see both retrieved context and generated answer"""
        # First, let's see what documents were retrieved
        retrieved_docs = self.retriever.get_relevant_documents(query)
        print("\nRetrieved Context:")
        for i, doc in enumerate(retrieved_docs, 1):
            print(f"\nDocument {i}:")
            print(doc.page_content[:200] + "...")

        # Now get the generated response
        response = self.qa_chain.invoke({"query": query})
        return response

def main():
    pdf_path = "/Users/yigitaydede/Dropbox/Documents/Papers/FertilityMC/TurkeyFertility/Papers/background/5.pdf"  # Replace with your PDF path
    rag = RAGSystem(pdf_path)

    question = "What does the document say about fertility?"
    response = rag.ask_question(question)
    
    print("\nGenerated Response:")
    print(response)

if __name__ == "__main__":
    main()