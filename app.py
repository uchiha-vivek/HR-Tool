import streamlit as st
import os
import tempfile
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

groq_api_key = st.secrets["GROQ_API_KEY"]
google_api_key = st.secrets["GOOGLE_API_KEY"]

# Set environment variables
os.environ["GROQ_API_KEY"] = groq_api_key
os.environ["GOOGLE_API_KEY"] = google_api_key
 
st.title("Gemma Model Document Q&A with Resume Upload and ATS Scoring")


llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")


prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    </context>
    Questions: {input}
    """
)


def embed_resumes(uploaded_files):
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        documents = []

        for uploaded_file in uploaded_files:
            if uploaded_file.type == "application/pdf":
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_file_path = temp_file.name

                
                loader = PyPDFLoader(temp_file_path)
                documents.extend(loader.load())

                
                os.remove(temp_file_path)

        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(documents)

        
        st.session_state.vectors = FAISS.from_documents(final_documents, st.session_state.embeddings)
        st.success("Resumes have been embedded successfully!")
    else:
        st.warning("Resumes are already embedded!")

# Function to calculate ATS score
def calculate_ats_score(job_description, resume_text):
    
    job_desc_embedding = st.session_state.embeddings.embed_documents([job_description])[0]
    resume_embedding = st.session_state.embeddings.embed_documents([resume_text])[0]


    similarity_score = cosine_similarity([job_desc_embedding], [resume_embedding])[0][0]
    
    
    return similarity_score


st.subheader("Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF resumes here", type=["pdf"], accept_multiple_files=True)

if st.button("Embed Resumes"):
    if uploaded_files:
        embed_resumes(uploaded_files)
    else:
        st.warning("Please upload at least one resume.")


st.subheader("Enter Job Description")
job_description = st.text_area("Enter job description for ATS scoring:")

if job_description:
    
    if st.button("Calculate ATS Scores"):
        if "vectors" in st.session_state:
            
            for uploaded_file in uploaded_files:
                if uploaded_file.type == "application/pdf":
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                        temp_file.write(uploaded_file.read())
                        temp_file_path = temp_file.name

                   
                    loader = PyPDFLoader(temp_file_path)
                    resume_text = ""
                    for doc in loader.load():
                        resume_text += doc.page_content

                    
                    ats_score = calculate_ats_score(job_description, resume_text)

                    
                    st.write(f"ATS Score for {uploaded_file.name}: {ats_score * 100:.2f}%")
                    
                     
                    os.remove(temp_file_path)
        else:
            st.warning("Please embed resumes first before calculating ATS scores.")
else:
    st.info("Please enter a job description for ATS scoring.")
