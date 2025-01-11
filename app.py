import streamlit as st
from prompts import *
from helpers import *
from screenshott import *
from PIL import Image
import pytesseract
from PyQt5.QtWidgets import QApplication
import tempfile
import os
import sys
from multiprocessing import Process
import pyautogui
from fpdf import FPDF

def save_text_as_temp_pdf(text):
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir="data")
        temp_path = temp_file.name
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, text.encode('latin-1', 'replace').decode('latin-1'))
        pdf.output(temp_path)
        return temp_path
    except Exception as e:
        print(f"Error creating temporary PDF file: {e}")
        return None

def start_screenshot():
    screenshot = pyautogui.screenshot()
    screenshot.save("screenshot.png")
    print("Capture d'écran terminée et sauvegardée sous 'screenshot.png'.")

def screenshot_notifier_process():
    app = QApplication(sys.argv)
    notifier = ScreenshotNotifier()
    notifier.show()
    notifier.start_countdown(on_complete=start_screenshot)
    sys.exit(app.exec_())

def show_screenshot_notifier():
    process = Process(target=screenshot_notifier_process)
    process.start()
    process.join()

def extract_text_from_image(image_path):
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        print(f"Erreur lors de l'extraction de texte : {e}")
        return ""

def main():
    st.title("ArChat : Chat With Scientific Papers")
    st.write("This application allows you to interact with scientific papers using RAG.")

    st.sidebar.header("Model Selection")
    model_choice = st.sidebar.selectbox("Select an LLM model:", ["Llama3.2 (3B)", "Google Gemma2 (2B)", "Microsoft Phi 3 Mini (3.8B)"])

    model_name = None
    if model_choice == "Llama3.2 (3B)":
        llm = OllamaLLM(model="llama3.2:3b")
        model_name = "llama3.2:3b"
    elif model_choice == "Google Gemma2 (2B)":
        llm = OllamaLLM(model="gemma2:2b")
        model_name = "gemma2:2b"
    elif model_choice == "Microsoft Phi 3 Mini (3.8B)":
        llm = OllamaLLM(model="phi3")
        model_name = "phi3"

    source_type = st.radio("Choose your source:", ["Upload a PDF", "Take a Screenshot"])

    if 'selected_text' not in st.session_state:
        st.session_state.selected_text = None
    if 'selected_pdf' not in st.session_state:
        st.session_state.selected_pdf = None

    if source_type == "Upload a PDF":
        if 'screenshot_path' in st.session_state:
            del st.session_state['screenshot_path']
        
        uploaded_file = st.file_uploader("Upload a new PDF", type=["pdf"])
        if uploaded_file is not None:
            new_pdf_name = save_uploaded_file(uploaded_file)
            st.session_state.selected_pdf = os.path.join("data", new_pdf_name)
            st.success(f"File uploaded: {new_pdf_name}")
            st.session_state.selected_text = get_txt_content(st.session_state.selected_pdf)

    elif source_type == "Take a Screenshot":
        if st.button("Take screenshot"):
            show_screenshot_notifier()
            screenshot_path = "screenshot.png"
            if os.path.exists(screenshot_path):
                st.session_state['screenshot_path'] = screenshot_path
                st.success("Capture d'écran effectuée avec succès.")
                extracted_text = extract_text_from_image(screenshot_path)
                if extracted_text.strip():
                    temp_pdf_path = save_text_as_temp_pdf(extracted_text)
                    if temp_pdf_path:
                        st.session_state.selected_pdf = temp_pdf_path
                        st.session_state.selected_text = extracted_text
                        st.success("Texte extrait et converti en PDF temporaire avec succès.")
                    else:
                        st.error("Erreur lors de la création du fichier PDF temporaire.")
                else:
                    st.warning("Aucun texte détecté dans la capture d'écran.")
            else:
                st.error("Erreur lors de la capture d'écran.")

    if st.session_state.get('screenshot_path'):
        st.image(st.session_state['screenshot_path'], caption="Capture d'écran", use_container_width=True)

    if not st.session_state.selected_text:
        st.warning("No source selected or text available.")
        return

    if "task_type" not in st.session_state:
        st.session_state.task_type = None

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Question Answering"):
            st.session_state.task_type = "Question answering"

    with col2:
        if st.button("Summary Generation"):
            st.session_state.task_type = "Summary Generation"

    with col3:
        if st.button("Reference Extraction"):
            st.session_state.task_type = "Reference Extraction"

    if st.session_state.task_type:
        st.write(f"Selected Task: {st.session_state.task_type}")

    if st.session_state.selected_pdf and st.session_state.task_type == "Question answering":
        question = st.text_input("Enter your question here:")
        if st.button("Submit"):
            if question.strip():
                vectorstore = load_pdf_to_vectorstore(st.session_state.selected_pdf)
                prompt = ChatPromptTemplate.from_template(question_answering_prompt)      
                relevent_docs = retrieve_from_vectorstore(vectorstore, query=question)
                prompt_with_context = prompt.format(context=relevent_docs, question=question)
                response = llm.invoke(prompt_with_context)

                st.success("Answer:")
                st.write(response.strip())
            else:
                st.warning("Please ask a question before submitting.")

    elif st.session_state.selected_pdf and st.session_state.task_type == "Summary Generation":
        if st.button("Generate Summary"):
            prompt = ChatPromptTemplate.from_template(summary_generation_template)      
            article_text = get_txt_content(st.session_state.selected_pdf)
            prompt_with_context = prompt.format(context=article_text)

            filename_with_ext = os.path.basename(st.session_state.selected_pdf)
            file_name_only = os.path.splitext(filename_with_ext)[0]
            output_pdf = file_name_only + "_summary.pdf"

            response = llm.invoke(prompt_with_context)

            markdown_to_pdf(response, output_file=output_pdf)
        
            full_pdf_path = os.path.join("summary", output_pdf)

            with open(full_pdf_path, "rb") as file:
                btn = st.download_button(
                    label="Download Summary PDF",
                    data=file,
                    file_name=output_pdf,
                    mime="text/pdf"
                )
            st.success("Summary generated! You can now download and open it in your browser.")

    elif st.session_state.task_type == "Reference Extraction":
        references = extract_references_with_prompts(st.session_state.selected_text, model_name)
        if references:
            filename_with_ext = os.path.basename(st.session_state.selected_pdf)
            file_name_only = os.path.splitext(filename_with_ext)[0]
            main_article_title = extract_pdf_title(st.session_state.selected_pdf)
            output_file = save_references_graph(main_article_title, references, file_name_only)

            st.success("References extracted successfully!")
            st.subheader("References Found:")

            with open(output_file, "rb") as f:
                st.download_button(
                    label="Download References Graph",
                    data=f,
                    file_name=file_name_only + "_graph.html",
                    mime="application/html",
                )
            
            st.json(references)
        else:
            st.warning('No References')
if __name__ == "__main__":
    main()
