import streamlit as st
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import Ollama

# If you wanted a small Hugging Face model instead of Ollama, you could do:
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
# from langchain.llms import HuggingFacePipeline
# model_name = "google/flan-t5-small"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
# pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512)
# llm = HuggingFacePipeline(pipeline=pipe)

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

# -----------------------------
# Using Ollama LLM (llama3.2 model)
# -----------------------------
llm = Ollama(model="llama3.2:1b")

# -----------------------------
# Prompt Templates
# -----------------------------
qa_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant. Given the following context from a PDF, please answer the question accurately. If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{question}
"""
)

question_generation_template = PromptTemplate(
    input_variables=["context"],
    template="""
You are an assistant tasked with generating relevant questions about the content below.

Content:
{context}

Please produce a list of 5 insightful questions that one might ask after reading this content.
"""
)

slide_generation_template = PromptTemplate(
    input_variables=["context"],
    template="""
You are an assistant that turns text into a concise slide deck summary. 
Summarize the main points of the content below into a structured set of bullet points 
as if creating slides for a presentation. Keep it concise and organized.

Content:
{context}
"""
)

# -----------------------------
# Main App
# -----------------------------
def main():
    st.title("PDF Query, Q&A, and Slides Generator")
    st.write("This application allows you to interact with PDFs using an Ollama-based LLM (llama3.2).")

    # --- PDF Selection ---
    st.sidebar.header("PDF Selection")
    pdf_files = get_available_pdfs()
    selected_pdf = st.sidebar.selectbox("Select a PDF:", pdf_files)
    
    uploaded_file = st.sidebar.file_uploader("Upload a new PDF", type=["pdf"])
    if uploaded_file is not None:
        new_pdf_name = save_uploaded_file(uploaded_file)
        st.sidebar.success(f"Uploaded {new_pdf_name}. Refresh the dropdown to see it.")
        pdf_files = get_available_pdfs()
        selected_pdf = new_pdf_name

    if selected_pdf:
        pdf_path = os.path.join("data", selected_pdf)
        text = load_pdf_text(pdf_path)

        # Split text into chunks if needed (optional)
        text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)
        context = "\n".join(chunks)

        st.write(f"**Selected PDF:** {selected_pdf}")

        # Feature 1: Q&A
        st.header("Ask a Question About the PDF")
        user_question = st.text_input("Enter your question:")
        if user_question:
            qa_chain = LLMChain(llm=llm, prompt=qa_template)
            answer = qa_chain.run(context=context, question=user_question)
            st.write("**Answer:**", answer.strip())

        # Feature 2: Generate Questions
        st.header("Generate Questions from the PDF")
        if st.button("Generate Questions"):
            qgen_chain = LLMChain(llm=llm, prompt=question_generation_template)
            questions = qgen_chain.run(context=context)
            st.write("**Generated Questions:**")
            st.write(questions.strip())

        # Feature 3: Generate Slides (Summaries)
        st.header("Generate Slides from the PDF")
        if st.button("Generate Slides"):
            slide_chain = LLMChain(llm=llm, prompt=slide_generation_template)
            slides = slide_chain.run(context=context)
            st.write("**Slide Summary:**")
            st.write(slides.strip())

if __name__ == "__main__":
    main()
