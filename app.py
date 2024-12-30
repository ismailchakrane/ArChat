import streamlit as st
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import Ollama
from langchain.chains.router.llm_router import RouterOutputParser
from langchain.chains.router import MultiPromptChain, LLMRouterChain
from langchain.text_splitter import RecursiveCharacterTextSplitter

import json



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

# -----------------------------
# Using Ollama LLM (llama3.2 model)
# -----------------------------
llm = Ollama(model="llama3.2:3b")

# -----------------------------
# Prompt Templates
# -----------------------------
question_answering_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""Use the following context to answer the question accurately and concisely.
    If the context does not contain relevant information, indicate "Information not available in the context."
    Context: {context}
    Question: {question}
    Answer (be clear and direct): """
)

question_generation_prompt = PromptTemplate(
    input_variables=["context", "num_questions"],
    template="""Generate exactly {num_questions} clear, concise, and relevant questions based solely on the provided context.
    Do not generate introductions, explanations, or non-interrogative sentences.
    Context: {context}
    Questions:
    """
)

training_plan_prompt = PromptTemplate(
    input_variables=["answers"],
    template="""Based on the following user responses, propose a structured training plan over several weeks.
    User responses: {answers}
    Training plan: """
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

# CHAINS SETUP
qa_chain = LLMChain(llm=llm, prompt=question_answering_prompt)
qg_chain = LLMChain(llm=llm, prompt=question_generation_prompt)
tp_chain = LLMChain(llm=llm, prompt=training_plan_prompt)

# Custom Output Parser
class JSONOutputParser(RouterOutputParser):
    """Parses output from the LLMRouterChain into the required dictionary format."""
    def parse(self, text: str) -> dict:
        try:
            # Attempt to parse as JSON
            return json.loads(text.strip())
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON output: {e}. Output was: {text}")

# Router Prompt with an Attached Output Parser
router_template = """
Determine the task to perform based on the user's input.
Input: {input}
The possible tasks are:
1. Answer a question
2. Generate evaluation questions
3. Training plan
Return **only** a strictly valid JSON object with the following two keys:
- "destination": A string corresponding to one of the possible tasks.
- "next_inputs": A dictionary containing the inputs needed to execute the chosen task.
Your response (only the valid JSON object):
"""

router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=JSONOutputParser()  # Attach the output parser
)

# Router Chain
router_chain = LLMRouterChain.from_llm(llm=llm, prompt=router_prompt)

# Chain Map for MultiPromptChain
chain_map = {
    "Answer a question": qa_chain,
    "Generate evaluation questions": qg_chain,
    "Training plan": tp_chain
}

multi_prompt_chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=chain_map,
    default_chain=qa_chain
)

# STREAMLIT INTERFACE

st.title("PDF Query, Q&A, training and Slides Generator")
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

PDF_FILE = os.path.join("data", selected_pdf)

st.write(f"**Selected PDF:** {selected_pdf}")
context = load_and_process_pdf(PDF_FILE)

task_type = st.radio(
    "choose a task :",
    ["Question answering", "Evaluation Questions Generating", "training plan Propose", "Slides Generating"]
)

# Stocker les réponses des utilisateurs
if "user_answers" not in st.session_state:
    st.session_state["user_answers"] = []

if task_type == "Question answering":
    question = st.text_input("Enter your question here :")
    if st.button("submit"):
        if question.strip():
            response = qa_chain.run(context=context, question=question)
            st.success("answer :")
            st.write(response.strip())
        else:
            st.warning("Please ask a question before submitting.")

elif task_type == "Evaluation Questions Generating":
    st.header("Generate questions based on the selected PDF")
    num_questions = st.number_input("questions :", min_value=1, max_value=10, value=5)
    
    # Initialiser les questions et réponses dans la session
    if "generated_questions" not in st.session_state:
        st.session_state["generated_questions"] = []
    if "user_answers" not in st.session_state or not isinstance(st.session_state["user_answers"], dict):
        st.session_state["user_answers"] = {}

    # Générer les questions
    if st.button("generate questions"):
        if context.strip():
            response = qg_chain.run(context=context, num_questions=num_questions)
            
            # Nettoyage des questions générées
            generated_questions = response.strip().split("\n")
            cleaned_questions = [
                q.strip() for q in generated_questions 
                if q.strip().endswith("?") and not q.lower().startswith("it seems that")
            ]
            
            # Limiter aux questions demandées
            limited_questions = cleaned_questions[:num_questions]

            if limited_questions:
                # Sauvegarder les questions générées dans la session
                st.session_state["generated_questions"] = limited_questions
                st.success("Questions successfully generated.")
            else:
                st.warning("No relevant question was generated. Try another context.")
        else:
            st.warning("Context is empty or irrelevant. Please check PDF.")
    
    # Afficher les questions générées et les réponses
    if st.session_state["generated_questions"]:
        st.header("Answer the questions generated:")
        for i, question in enumerate(st.session_state["generated_questions"], start=1):
            st.write(f"**Question {i}:** {question}")
            
            # Récupérer la réponse existante ou laisser vide
            key = f"response_{i}"
            default_answer = st.session_state["user_answers"].get(key, "")
            
            # Zone de texte pour la réponse
            user_answer = st.text_area(f"Your answer to the question {i} :", value=default_answer, key=key)
            
            # Mettre à jour la réponse dans la session
            st.session_state["user_answers"][key] = user_answer.strip()

    # Option pour afficher toutes les réponses mémorisées
    if st.button("See all recorded ansewrs"):
        st.header("Réponses mémorisées :")
        for i, question in enumerate(st.session_state["generated_questions"], start=1):
            answer = st.session_state["user_answers"].get(f"response_{i}", "Pas encore de réponse")
            st.write(f"**Question {i}:** {question}")
            st.write(f"**Answer :** {answer}")
elif task_type == "training plan Propose":
    st.header("training plan Propose")

    # Vérifier s'il y a des réponses enregistrées
    if "user_answers" in st.session_state and st.session_state["user_answers"]:
        # Récupérer toutes les réponses enregistrées
        all_answers = "\n".join([
            f"Question {i + 1}: {st.session_state['generated_questions'][i]}\nRéponse: {answer}"
            for i, answer in enumerate(st.session_state["user_answers"].values())
        ])
        
        # Générer le plan de formation
        training_plan = tp_chain.run(answers=all_answers)
        st.success("Proposed training plan :")
        st.write(training_plan.strip())
    else:
        st.warning("No user response available. Please answer the questions generated before proposing a training plan.")

        
    st.header("Slides Generating")
    if st.button("Slides Generating"):
            slide_chain = LLMChain(llm=llm, prompt=slide_generation_template)
            slides = slide_chain.run(context=context)
            st.write("**Slide Summary:**")
            st.write(slides.strip())