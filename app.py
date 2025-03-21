import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time  # For progress bar

# Function to extract text from PDF
def extract_text_from_pdf(file):
    try:
        pdf = PdfReader(file)
        text = "" 
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text() + " "
        return text.strip() if text else "Error: Could not extract text."
    except Exception as e:
        return f"Error: {e}"

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer(stop_words="english").fit_transform(documents)
    vectors = vectorizer.toarray()

    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]

    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
    return cosine_similarities

# Custom CSS Styling
st.markdown(
    """
    <style>
    .stApp {
        background-color:#6487de; /* Unified background color */
        font-family:'Arial';
    }
    [data-testid="stSidebar"] {
        background-color: #2E86C1; 
        color: white;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: white;
    }
    div.stButton > button {
        background-color: #FF5733;
        color: white;
        font-size: 16px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar Navigation
with st.sidebar:
    st.image("https://cdn.prod.website-files.com/64d03d94c73469cb85a2d02f/64d03d94c73469cb85a2d3ca_shutterstock_1279483576.png", width=200)
    st.header("Navigation")
    page = st.radio("Go to:", ["Home", "Rank Resume"])

# **Home Page**
if page == "Home":
    st.markdown(
        """
        <div style="text-align: center;">
            <h1 style="color: #333;">Welcome to the AI Resume Ranking System!</h1>
            <h3 style="color: #333;">Helping recruiters <b>automatically rank resumes</b> based on job descriptions.</h3>
            <ul style="text-align: left;">
                <li> <b><i>Upload multiple PDF resumes</i></b></li>
                <li> <b><i>Get AI-powered relevance scores</i></b></li>
                <li> <b><i>Download ranked results as CSV</i></b></li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

# **Rank Resume Page**
else:
    st.title("AI Resume Screening & Candidate Ranking System")

    # Job description input
    st.header("Job Description")
    job_description = st.text_area("Enter the job description")

    # File uploader (NOW INSIDE the "Rank Resume" page)
    st.header("Upload Resumes")
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

    if st.button("Upload"):
        if uploaded_files and job_description:
            st.subheader("Processing Resumes...")
            
            progress_bar = st.progress(0)
            for percent in range(0, 101, 10):
                time.sleep(0.2)
                progress_bar.progress(percent)

            resume_texts = [extract_text_from_pdf(file) for file in uploaded_files]
            valid_resumes = [text for text in resume_texts if not text.startswith("Error")]

            if valid_resumes:
                scores = rank_resumes(job_description, valid_resumes)
                ranked_resumes = sorted(zip(uploaded_files, scores), key=lambda x: x[1], reverse=True)

                # Display results
                results_df = pd.DataFrame(
                    [(file.name, round(score * 100, 2)) for file, score in ranked_resumes], 
                    columns=["Resume", "Relevance Score (%)"]
                )

                st.subheader("Ranked Resumes")
                st.dataframe(results_df.style.highlight_max(axis=0))  # Highlight top resume

            else:
                st.error(" No valid resumes found. Please check the uploaded files.")
        else:
            st.warning("Please enter a job description and upload resumes before ranking.")
