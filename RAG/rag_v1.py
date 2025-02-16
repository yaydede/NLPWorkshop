# Initial and simple version of the RAG model
# The document is loaded from a PDF file and the user can ask questions about the document
# The document is processed using OpenAI embeddings and a FAISS vector store
# The user can ask questions about the document and receive answers based on the retrieved information
# The vector store can be saved for future use


import os
import io
import openai
from langchain_community.document_loaders import PyPDFLoader, PDFPlumberLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI  # Changed from OpenAI to ChatOpenAI

# Set the API key as an environment variable
os.environ["OPENAI_API_KEY"] = "sk-xFUOUX2O2-PYxkPAeeILN6QY_kLPS7jEf0wg898XMtT3BlbkFJgNF3p3cbgEbhfuoWIROHbHlrbw1aIeSNb_CuRln3MA"

def load_pdf(pdf_source):
    if isinstance(pdf_source, str):
        if not os.path.exists(pdf_source):
            raise FileNotFoundError(f"PDF file not found at {pdf_source}")
        return PyPDFLoader(pdf_source).load()
    else:
        # Handle Streamlit's UploadedFile object
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_source.getvalue())
            tmp_file.flush()
            documents = PyPDFLoader(tmp_file.name).load()
            os.unlink(tmp_file.name)  # Clean up the temporary file
            return documents

def create_vector_store(documents):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(documents, embeddings)

def get_response(query, retriever):
    llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return qa_chain.invoke({"query": query})

# Ensure these functions are available for import
__all__ = ['load_pdf', 'create_vector_store', 'get_response']

if __name__ == "__main__":
    pdf_path = "/Users/yigitaydede/Dropbox/Documents/Papers/FertilityMC/TurkeyFertility/Papers/background/5.pdf"
    documents = load_pdf(pdf_path)
    vector_store = create_vector_store(documents)
    retriever = vector_store.as_retriever()
    query = "What does the document say about fertility?"
    response = get_response(query, retriever)
    print("Response:", response)
