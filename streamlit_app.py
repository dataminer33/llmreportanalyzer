import streamlit as st
import pandas as pd
import base64, os, re
from utils.constants import *
from utils.pdf_qa import PdfQA
import sys



# Streamlit app code
st.set_page_config(
    page_title='Report Analysis Tool',
    page_icon='🧮',
    layout='wide',
    initial_sidebar_state='auto',
)


@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64("static/images/background_vq.png")


page_bg_img = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

body {{
    font-family: 'Poppins', sans-serif;
    color: #ffffff;
    background-color: #001a33;
}}

[data-testid="stAppViewContainer"] > .main {{
    background-image: url("data:image/png;base64,{img}");
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    background-size: cover;
}}

[data-testid="stSidebar"] > div:first-child {{
       background-color: #002b4d;
        background-image: linear-gradient(315deg, #002b4d 0%, #001a33 74%);
}}

.stApp {{
    background-color: rgba(255, 255, 255, 0.7);
}}

h1, h2, h3 {{
    color: #00bfff;
}}

.stButton>button {{
    background-color: #00bfff;
    color: #001a33;
    border: none;
    border-radius: 4px;
    padding: 0.5rem 1rem;
    font-weight: 600;
    transition: all 0.3s ease;
}}

.stButton>button:hover {{
    background-color: #0099cc;
    box-shadow: 0 0 15px rgba(0,191,255,0.5);
}}

.stTextInput>div>div>input {{
    background-color: #002b4d;
    color: #ffffff;
    border: 1px solid #00bfff;
    border-radius: 4px;
}}

.custom-info-box {{
    background-color: #003366;
    border-left: 6px solid #00bfff;
    margin-bottom: 15px;
    padding: 16px;
    border-radius: 4px;
}}

.stTabs [data-baseweb="tab-list"] {{
    gap: 24px;
    border-bottom: 1px solid #00bfff;
}}

.stTabs [data-baseweb="tab"] {{
    height: 60px;
    white-space: pre-wrap;
    background-color: transparent !important;
    border-radius: 4px 4px 0 0;
    gap: 1px;
    padding-top: 10px;
    padding-bottom: 10px;
    color: #ffffff;
}}

.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {{
    font-size: 24px;
    font-weight: 600;
}}

.stTabs [data-baseweb="tab-highlight"] {{
    background-color: #00bfff  !important;
}}

.stTabs [data-baseweb="tab"][aria-selected="true"] {{
    color: #00bfff  !important;
}}
[data-testid="stHeader"] {{
    background-color: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
    right: 2rem;
}}
.stSelectbox select {{
    background-color: #002b4d;
    color: #ffffff;
    border: 1px solid #00bfff;
}}
.stCheckbox label {{
    color: #ffffff;
}}
.stDataFrame {{
    background-color: #002b4d;
    color: #ffffff;
}}

.stDataFrame [data-testid="stTable"] {{
    background-color: #003366;
}}

p, li, span {{
    color: #ffffff;
}}

[data-testid="stFileUploader"] {{
    background-color: #002b4d;
    border: 1px dashed #00bfff;
    border-radius: 4px;
    padding: 10px;
}}

.stAlert {{
    background-color: #003366;
    color: #ffffff;
    border-left-color: #00bfff;
}}
</style>
"""


st.markdown(page_bg_img, unsafe_allow_html=True)



if "pdf_qa_model" not in st.session_state:
    st.session_state["pdf_qa_model"]:PdfQA = PdfQA(openai_api_key=OPENAI_API_KEY, huggingface_api_key=HUGGINGFACE_API_KEY)  # Initialisation

@st.cache_resource
def load_llm(llm):
    if (llm == LLM_OPENAI_GPT35) or (llm == LLM_OPENAI_GPT4O) or (llm == LLM_OPENAI_GPT4O_MINI) or (llm == LLM_OPENAI_GPT4):
        pass
    elif llm == LLM_LLAMA3_INSTRUCT:
        return PdfQA.create_llama3_8B_instruct()
    else:
        raise ValueError("Invalid LLM setting")

@st.cache_resource
def load_emb(emb):
    if emb == EMB_GTE_BASE:
        return PdfQA.create_mpnet_base_v1()
    else:
        raise ValueError("Invalid embedding setting")


def categorize_response(response):
    response_lower = response.lower()
    if re.search(r'\byes\b', response_lower):
        return "Yes"
    elif re.search(r'\bno\b', response_lower):
        return "No"
    else:
        return "Not Given"


st.title("Report Analysis Tool")

with st.sidebar:
    st.header("Configuration")


    emb = EMB_GTE_BASE
    llm = st.radio("**Select LLM Model**", [LLM_OPENAI_GPT35,LLM_OPENAI_GPT4O,LLM_OPENAI_GPT4O_MINI,LLM_OPENAI_GPT4,LLM_LLAMA3_INSTRUCT], index=4)
    pdf_file = st.file_uploader("**Upload PDF**", type="pdf")

    if st.button("Submit") and pdf_file is not None:
        with st.spinner(text="Processing PDF and Generating Embeddings.."):
            try:

                pdf_path = os.path.join(os.path.dirname(__file__), pdf_file.name)
                
                with open(pdf_path, "wb") as f:
                    f.write(pdf_file.getbuffer())
                
                st.session_state["pdf_qa_model"].config = {
                    "pdf_path": pdf_path,
                    "embedding": emb,
                    "llm": llm
                } 
                st.session_state["pdf_qa_model"].init_embeddings()
                st.session_state["pdf_qa_model"].init_models()
                st.session_state["pdf_qa_model"].vector_db_pdf()
                st.session_state["pdf_qa_model"].retreival_qa_chain()
                st.sidebar.success("PDF processed successfully")
            except Exception as e:
                st.error(f"An error occurred: {e}")



if "pdf_file_name" in st.session_state:
    st.write(f"Currently loaded PDF: {st.session_state['pdf_file_name']}")

# Create two tabs
tab1, tab2 = st.tabs(["Batch Q&A", "Interactive Q&A"])

# Tab 1: Batch Q&A
with tab1:
    st.header("Batch Q&A")
    questions_file = st.file_uploader("Upload a CSV file with questions", type="csv")
    
    if questions_file is not None:
        questions_df = pd.read_csv(questions_file,encoding='unicode_escape')
        st.write("Preview of uploaded questions:")
        st.write(questions_df.head())
        
        if st.button("Process Batch Questions"):
            if "pdf_qa_model" in st.session_state and hasattr(st.session_state["pdf_qa_model"], "answer_query"):
                answers = []
                relevant_pages  = []
                for question in questions_df['Questions']:
                
                    result = st.session_state["pdf_qa_model"].answer_query(st, question + 'Give me an answer: yes or no answer and reasoning from the context')
                    answers.append(result["result"])
                    relevant_pages.append([doc.metadata.get("page", None) for doc in result["source_documents"]])

                questions_df['Response'] = answers
                questions_df['Relevant_pages'] = list(relevant_pages)
                questions_df["Answer_model"] = questions_df["Response"].apply(categorize_response)
                
                st.write("Results:")
                st.write(questions_df)
                
                # Option to download results
                csv = questions_df.to_csv(index=False)


                st.download_button(
                    label="Download results as CSV",
                    data=csv,
                    file_name="qa_results.csv",
                    mime="text/csv",
                )
            else:
                st.error("Please upload and process a PDF file first.")

# Tab 2: Interactive Q&A
with tab2:
    st.header("Interactive Q&A")
    question = st.text_input('Ask a question', 'What is this document?')
    st.write(" ")
    if st.button("Answer"):
        try:
            st.session_state["pdf_qa_model"].retreival_qa_chain()
            answer = st.session_state["pdf_qa_model"].answer_query(st, question)
        except Exception as e:
            st.error(f"Error answering the question: {str(e)}")

key_features = [
    "Efficient document processing and analysis",
    "Advanced natural language understanding",
    "Customizable question-answering system",
    "Batch processing for multiple queries",
    "Integration with state-of-the-art language models"
]

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔑 Key Features")
for feature in key_features:
    st.sidebar.markdown(f"- {feature}")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center;'>
        <p>Developed by Alexandre Da Silva | © 2024 All Rights Reserved</p>
        <p>Empowering businesses with intelligent document analysis</p>
    </div>
    """, unsafe_allow_html=True)
