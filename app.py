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
    """
    Capture l'écran et enregistre l'image.
    """
    import pyautogui
    screenshot = pyautogui.screenshot()
    screenshot.save("screenshot.png")
    print("Capture d'écran terminée et sauvegardée sous 'screenshot.png'.")


from multiprocessing import Process

def screenshot_notifier_process():
    """
    Processus de notification de capture d'écran avec PyQt5.
    """
    app = QApplication(sys.argv)
    notifier = ScreenshotNotifier()
    notifier.show()
    notifier.start_countdown(on_complete=start_screenshot)
    sys.exit(app.exec_())

def show_screenshot_notifier():
    """
    Lance le processus de notification.
    """
    process = Process(target=screenshot_notifier_process)
    process.start()
    process.join()  # Attendre que le processus se termine


def extract_text_from_image(image_path):
    """
    Extrait le texte d'une image (capture d'écran).
    """
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        print(f"Erreur lors de l'extraction de texte : {e}")
        return ""


def main():
    # Initialiser l'état de session
    if 'selected_text' not in st.session_state:
        st.session_state['selected_text'] = None

    st.title("ArChat : Chat With Scientific Papers or Screenshots")
    st.write("This application allows you to interact with scientific papers or screenshots using a RAG-System.")

    # Modèle de langage
    st.sidebar.header("Model Selection")
    model_choice = st.sidebar.selectbox("Select an LLM model:", ["Llama3.2 (3B)", "Google Gemma2 (2B)", "Microsoft Phi 3 Mini (3.8B)"])
    if model_choice == "Llama3.2 (3B)":
        llm = OllamaLLM(model="llama3.2:3b")
    elif model_choice == "Google Gemma2 (2B)":
        llm = OllamaLLM(model="gemma2:2b")
    elif model_choice == "Microsoft Phi 3 Mini (3.8B)":
        llm = OllamaLLM(model="phi3")

    # Sélection de la source
    source_type = st.radio("Choisissez la source :", ["Téléverser un fichier PDF", "Prendre une capture d'écran"])

    if source_type == "Téléverser un fichier PDF":
        # Téléversement de fichier
        uploaded_file = st.sidebar.file_uploader("Téléverser un fichier PDF", type=["pdf"])
        if uploaded_file is not None:
            new_pdf_name = save_uploaded_file(uploaded_file)
            st.sidebar.success(f"Fichier téléversé : {new_pdf_name}.")
            pdf_path = os.path.join("data", new_pdf_name)
            st.write(f"**Fichier sélectionné :** {pdf_path}")
            st.session_state['selected_text'] = get_txt_content(pdf_path)

    elif source_type == "Prendre une capture d'écran":
        # Capture d'écran
        if st.button("Prendre une capture d'écran"):
            show_screenshot_notifier()
            screenshot_path = "screenshot.png"
            if os.path.exists(screenshot_path):
                st.success("Capture d'écran effectuée avec succès.")
                extracted_text = extract_text_from_image(screenshot_path)
                if extracted_text.strip():
                    st.session_state['selected_text'] = extracted_text  # Stocker le texte dans session_state
                    st.write("Texte extrait de la capture d'écran :")
                    st.text(extracted_text)
                else:
                    st.warning("Aucun texte détecté dans la capture d'écran.")
            else:
                st.error("Erreur lors de la capture d'écran.")

    # Récupération du texte sélectionné
    selected_text = st.session_state.get('selected_text', None)

    if selected_text:
        task_type = st.radio(
            "Choisissez une tâche :",
            ["Question answering", "Summary Generation", "References Graph Generation", "Reference Extraction"]
        )

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
        # Ajoutez d'autres fonctionnalités comme Reference Extraction ici...
    else:
        st.warning("Aucune source sélectionnée ou texte disponible.")



if __name__ == "__main__":
    main()
