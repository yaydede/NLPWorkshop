import streamlit as st
from rag_v2 import RAGSystem
import tempfile
import os
from typing import List

# Page config and styling
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="📚",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
    }
    .upload-status {
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("📚 RAG Assistant")
st.markdown("---")

# Main container
main_container = st.container()
with main_container:
    # Document upload section
    st.subheader("📄 Document Upload")
    st.markdown("Upload PDF files for analysis (maximum 10 files)")

    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

    # Show upload status
    if uploaded_files:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(
                f"""
                <div class="upload-status" style="background-color: #E8F5E9;">
                    📎 Files selected: {len(uploaded_files)}
                </div>
                """,
                unsafe_allow_html=True
            )
            # Display file names
            for file in uploaded_files:
                st.markdown(f"- {file.name}")
        with col2:
            process_button = st.button("Process Documents 🚀")
    
    if uploaded_files and process_button:
        if len(uploaded_files) > 10:
            st.error("Please upload a maximum of 10 PDF files.")
        else:
            try:
                with st.spinner("Processing documents..."):
                    # Create temporary files for all uploaded PDFs
                    temp_paths: List[str] = []
                    for uploaded_file in uploaded_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            temp_paths.append(tmp_file.name)

                    # Initialize RAG system with all temporary files
                    rag_system = RAGSystem(temp_paths)
                    st.session_state.rag_system = rag_system
                    
                    # Save the vector store
                    rag_system.save_vector_store()
                    
                    # Clean up temporary files
                    for temp_path in temp_paths:
                        os.unlink(temp_path)
                    
                    st.success(f"✨ {len(uploaded_files)} documents processed and ready for questions!")
            except Exception as e:
                st.error(f"Error: {e}")

    # Query section
    if 'rag_system' in st.session_state:
        st.markdown("---")
        st.subheader("❓ Ask Questions")
        query = st.text_input("What would you like to know about the documents?", 
                            placeholder="Enter your question here...")

        if st.button("Get Answer 🔍"):
            with st.spinner("Generating response..."):
                response = st.session_state.rag_system.ask_question(query)["result"]
                
                # Display response in a nice container
                st.markdown("### 🤖 Response")
                st.markdown(
                    f"""
                    <div style="background-color: #F5F5F5; padding: 20px; border-radius: 5px;">
                    {response}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    else:
        st.error("Please load documents first.")
