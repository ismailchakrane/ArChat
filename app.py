import streamlit as st
from streamlit_tags import st_tags
import os
from prompts import *
from helpers import *
from screenshott import *
import json
from PIL import Image
import pytesseract
from PyQt5.QtWidgets import QApplication
import sys


def start_screenshot():
    import pyautogui
    screenshot = pyautogui.screenshot()
    screenshot.save("screenshot.png")
    print("Capture d'écran terminée et sauvegardée sous 'screenshot.png'.")


from multiprocessing import Process

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
    if 'selected_text' not in st.session_state:
        st.session_state['selected_text'] = None
    if 'last_task' not in st.session_state:
        st.session_state['last_task'] = None  # Suivi de la dernière tâche effectuée

    st.title("ArChat : Chat With Scientific Papers or Screenshots")
    st.write("This application allows you to interact with scientific papers or screenshots using a RAG-System.")

    st.sidebar.header("Model Selection")
    model_choice = st.sidebar.selectbox("Select an LLM model:", ["Llama3.2 (3B)", "Google Gemma2 (2B)", "Microsoft Phi 3 Mini (3.8B)"])
    llm = None
    if model_choice == "Llama3.2 (3B)":
        llm = OllamaLLM(model="llama3.2:3b")
    elif model_choice == "Google Gemma2 (2B)":
        llm = OllamaLLM(model="gemma2:2b")
    elif model_choice == "Microsoft Phi 3 Mini (3.8B)":
        llm = OllamaLLM(model="phi3")

    source_type = st.radio("Choisissez la source :", ["Select a PDF file", "Take screenshot"])

    if source_type == "Select a PDF file":
        # Réinitialiser le chemin de la capture d'écran lorsque l'utilisateur choisit cette option
        if 'screenshot_path' in st.session_state:
            del st.session_state['screenshot_path']
        
        uploaded_file = st.sidebar.file_uploader("Select a PDF file", type=["pdf"])
        if uploaded_file is not None:
            new_pdf_name = save_uploaded_file(uploaded_file)
            st.sidebar.success(f"File selected : {new_pdf_name}.")
            pdf_path = os.path.join("data", new_pdf_name)
            st.write(f"**File selected :** {pdf_path}")
            st.session_state['selected_text'] = get_txt_content(pdf_path)

    elif source_type == "Take screenshot":
        if st.button("Take screenshot"):
            show_screenshot_notifier()
            screenshot_path = "screenshot.png"
            if os.path.exists(screenshot_path):
                st.session_state['screenshot_path'] = screenshot_path  # Stocker le chemin de l'image
                st.success("Capture d'écran effectuée avec succès.")
                extracted_text = extract_text_from_image(screenshot_path)
                if extracted_text.strip():
                    st.session_state['selected_text'] = extracted_text
                else:
                    st.warning("Aucun texte détecté dans la capture d'écran.")
            else:
                st.error("Erreur lors de la capture d'écran.")

    # Affichage persistant de l'image
    if st.session_state.get('screenshot_path'):
        st.image(st.session_state['screenshot_path'], caption="Capture d'écran", use_column_width=True)


    selected_text = st.session_state.get('selected_text', None)

    if selected_text:
        task_type = st.radio(
            "Choisissez une tâche :",
            ["Question answering", "Summary Generation", "References Graph Generation", "Reference Extraction"]
        )

        # Réinitialiser l'état de la tâche si une nouvelle est sélectionnée
        if st.session_state['last_task'] != task_type:
            st.session_state['last_task'] = task_type
            st.session_state['task_output'] = None

        if task_type == "Summary Generation":
            if st.button("Générer un résumé"):
                prompt = ChatPromptTemplate.from_template(summary_generation_template)
                prompt_with_context = prompt.format(context=selected_text)

                output_pdf = "summary_output.pdf"
                response = llm.invoke(prompt_with_context)
                markdown_to_pdf(response, output_file=output_pdf)

                with open(os.path.join("summary", output_pdf), "rb") as file:
                    st.download_button(
                        label="Télécharger le résumé en PDF",
                        data=file,
                        file_name=output_pdf,
                        mime="text/pdf",
                    )
                st.success("Résumé généré avec succès !")

        elif task_type == "References Graph Generation":
            if st.button("Générer le graphe des références"):
                reference_graph = extract_references_from_article(selected_text)
                graph_path = reference_graph_to_html(reference_graph)
                st.success("Graphe des références généré avec succès !")
                st.write(f"[Voir le graphe des références]({graph_path})")

        elif task_type == "Reference Extraction":
            if st.button("Extraire les références"):
                references = extract_references_with_prompts(selected_text)
                references_file = save_references_to_json(references)
                with open(references_file, "rb") as file:
                    st.download_button(
                        label="Télécharger les références en JSON",
                        data=file,
                        file_name=references_file,
                        mime="application/json",
                    )
                st.success("Références extraites avec succès !")

        elif task_type == "Question answering":
            question = st.text_input("Posez votre question :")
            if st.button("Obtenir une réponse"):
                if question.strip():
                    prompt = ChatPromptTemplate.from_template(question_answering_prompt)
                    prompt_with_context = prompt.format(context=selected_text, question=question)
                    response = llm.invoke(prompt_with_context)
                    st.write("Réponse :")
                    st.write(response)
                else:
                    st.warning("Veuillez entrer une question.")
    else:
        st.warning("Aucune source sélectionnée ou texte disponible.")


if __name__ == "__main__":
    main()
