import os
import pickle
import streamlit as st
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import pipeline
from PyPDF2 import PdfReader
from dotenv import load_dotenv


load_dotenv()


HF_API_KEY = os.getenv("HF_API_KEY")
os.environ["HUGGINGFACE_API_KEY"] = HF_API_KEY  


model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)


qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")


file_path = "faiss_store_hf.pkl"


st.title("ReadBot: PDF Research Tool 📖")
st.sidebar.title("Upload PDF Files")


uploaded_files = st.sidebar.file_uploader(
    "Upload one or more PDF files", accept_multiple_files=True, type=["pdf"]
)

process_files_clicked = st.sidebar.button("Process Files")

main_placeholder = st.empty()

if process_files_clicked:
    if not uploaded_files:
        st.error("Please upload at least one PDF file.")
    else:
        with st.spinner("Processing PDFs and creating embeddings..."):
            all_text = ""
           
            for file in uploaded_files:
                reader = PdfReader(file)
                for page in reader.pages:
                    all_text += page.extract_text()

          
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000
            )
            docs = text_splitter.create_documents([all_text])

           
            vectorstore_hf = FAISS.from_documents(docs, embeddings)

           
            with open(file_path, "wb") as f:
                pickle.dump(vectorstore_hf, f)
        
        st.success("Processing complete!")


query = main_placeholder.text_input("Question: ")

if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            retriever = vectorstore.as_retriever()
            docs = retriever.get_relevant_documents(query)
            
           
            context = " ".join([doc.page_content for doc in docs])
            result = qa_pipeline(question=query, context=context)
            
            
            st.header("Answer")
            st.write(result.get("answer", "No answer available."))

            
    else:
        st.error("FAISS index file not found. Please process files first.")


