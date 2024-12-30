import streamlit as st
import os
import speech_recognition as sr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import Ollama
from langchain.chains.router.llm_router import RouterOutputParser
from langchain.chains.router import MultiPromptChain, LLMRouterChain

from utils.prompts import question_answering_prompt, question_generation_prompt, training_plan_prompt, slide_generation_template

# -----------------------------
# Utility Functions
# -----------------------------
def load_pdf_text(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text = "\n".join([doc.page_content for doc in docs])
    return text

def get_available_pdfs(directory="data"):
    files = [f for f in os.listdir(directory) if f.lower().endswith(".pdf")]
    return files

def save_uploaded_file(uploaded_file, directory="data"):
    with open(os.path.join(directory, uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    return uploaded_file.name

# Charger et traiter le contenu du PDF
def load_and_process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Suppression des doublons
    unique_documents = {doc.page_content: doc for doc in documents}
    documents = list(unique_documents.values())

    # Diviser en chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return "\n".join([chunk.page_content for chunk in chunks])

# Fonction pour la reconnaissance vocale
def recognize_audio():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        st.info("Listening for your question...")
        audio = recognizer.listen(source)
    
    try:
        question = recognizer.recognize_google(audio)
        st.success(f"Question recognized: {question}")
        return question
    except sr.UnknownValueError:
        st.error("Sorry, I couldn't understand the audio. Please try again.")
        return None
    except sr.RequestError:
        st.error("Could not request results from the speech recognition service.")
        return None

# -----------------------------
# Initialize LLM
# -----------------------------
st.title("PDF Query, Q&A, Training, and Slides Generator")
st.write("This application allows you to interact with PDFs using various LLM models.")

# --- Model Selection ---
st.sidebar.header("Model Selection")
model_choice = st.sidebar.selectbox("Select an LLM model:", ["Ollama (Llama3.2)", "Google Gemma2 (2B)", "Microsoft Phi 3 Mini (3.8B)"])

# Initialize LLM based on the selected model
if model_choice == "Ollama (Llama3.2)":
    llm = Ollama(model="llama3.2:3b")
elif model_choice == "Google Gemma2 (2B)":
    llm = Ollama(model="gemma2:2b")
elif model_choice == "Microsoft Phi 3 Mini (3.8B)":
    llm = Ollama(model="phi3")

# CHAINS SETUP
# CHAINS SETUP
qa_chain = LLMChain(llm=llm, prompt=question_answering_prompt)
qg_chain = LLMChain(llm=llm, prompt=question_generation_prompt)
tp_chain = LLMChain(llm=llm, prompt=training_plan_prompt)

# STREAMLIT INTERFACE CONTINUED
st.sidebar.header("PDF Selection")
pdf_files = get_available_pdfs()
selected_pdf = st.sidebar.selectbox("Select a PDF:", pdf_files)

uploaded_file = st.sidebar.file_uploader("Upload a new PDF", type=["pdf"])
if uploaded_file is not None:
    new_pdf_name = save_uploaded_file(uploaded_file)
    st.sidebar.success(f"Uploaded {new_pdf_name}. Refresh the dropdown to see it.")
    pdf_files = get_available_pdfs()
    selected_pdf = new_pdf_name

PDF_FILE = os.path.join("data", selected_pdf)

st.write(f"**Selected PDF:** {selected_pdf}")
context = load_and_process_pdf(PDF_FILE)

task_type = st.radio(
    "Choose a task:",
    ["Question answering", "Evaluation Questions Generating", "Training plan Propose", "Slides Generating"]
)

if task_type == "Question answering":
    input_method = st.radio("Choose input method", ["Text Q&A", "Audio Q&A"])

    if input_method == "Text Q&A":
        question = st.text_input("Enter your question here:")
        if st.button("Submit"):
            if question.strip():
                response = qa_chain.run(context=context, question=question)
                st.success("Answer:")
                st.write(response.strip())
            else:
                st.warning("Please ask a question before submitting.")

    elif input_method == "Audio Q&A":
        if st.button("Ask by Audio"):
            question = recognize_audio()  # Capture and recognize the audio question
            if question:
                response = qa_chain.run(context=context, question=question)
                st.success("Answer:")
                st.write(response.strip())

elif task_type == "Evaluation Questions Generating":
    num_questions = st.number_input("Number of questions:", min_value=1, max_value=10, value=5)
    if st.button("Generate questions"):
        response = qg_chain.run(context=context, num_questions=num_questions)
        st.success("Generated Questions:")
        st.write(response.strip())

elif task_type == "Training plan Propose":
    if st.button("Propose Training Plan"):
        training_plan = tp_chain.run(answers=context)
        st.success("Training Plan:")
        st.write(training_plan.strip())

elif task_type == "Slides Generating":
    slide_chain = LLMChain(llm=llm, prompt=slide_generation_template)
    slides = slide_chain.run(context=context)
    st.success("Slide Summary:")
    st.write(slides.strip())
