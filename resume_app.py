

import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Streamlit page config
st.set_page_config(page_title="AI Resume Screening", page_icon="📄", layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
        .big-font { font-size:25px !important; font-weight: bold; }
        .score-bar { height: 8px; border-radius: 5px; background: #f0f0f0; }
        .score-fill { height: 8px; border-radius: 5px; background: #4CAF50; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title
st.markdown('<p class="big-font">📄 AI Resume Screening & Candidate Ranking</p>', unsafe_allow_html=True)
st.write("Upload resumes and get ranked results based on the job description.")

# Extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""
    return text.strip()

# Resume Ranking Function
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()

    # Calculate cosine similarity
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity(job_description_vector.reshape(1, -1), resume_vectors).flatten()
    return cosine_similarities

# Layout - Job Description Input
st.header("📝 Job Description")
job_description = st.text_area("Enter the job description here...", height=150)

# File Upload Section
st.header("📂 Upload Resumes")
uploaded_files = st.file_uploader(
    "Upload PDF resumes (multiple files supported)", 
    type=["pdf"], accept_multiple_files=True
)

# Processing resumes if uploaded
if uploaded_files and job_description:
    st.header("📊 Resume Ranking Results")

    resumes = [extract_text_from_pdf(file) for file in uploaded_files if extract_text_from_pdf(file)]
    
    if resumes:
        scores = rank_resumes(job_description, resumes)

        # Create DataFrame
        results = pd.DataFrame({
            "Resume": [file.name for file in uploaded_files],
            "Score": scores
        }).sort_values(by="Score", ascending=False)

        # Display Results
        for index, row in results.iterrows():
            st.subheader(f"📜 {row['Resume']}")
            st.markdown(f'<div class="score-bar"><div class="score-fill" style="width:{row["Score"]*100}%"></div></div>', unsafe_allow_html=True)
            st.write(f"🔹 **Match Score:** {row['Score']:.2%}")

        # Allow users to download results
        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Download Results as CSV", data=csv, file_name="resume_ranking_results.csv", mime="text/csv")

    else:
        st.warning("No valid text extracted from the uploaded resumes.")
