import pdfplumber
import google.generativeai as genai
import gradio as gr

# Configurez votre clé API
genai.configure(api_key="")

# Étape 1 : Extraction de texte depuis un PDF
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Étape 2 : Découpage du texte en segments
def split_text_into_chunks(text, chunk_size=200):
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Étape 3 : Génération de réponse avec Gemini
def generate_response_gemini(context, question):
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    response = model.generate_content(prompt)
    return response.text if hasattr(response, "text") else "Pas de réponse générée."

# Pipeline complet
def process_question(pdf_path, question):
    # Extraire le texte depuis le PDF
    text = extract_text_from_pdf(pdf_path.name)
    
    # Diviser en segments
    chunks = split_text_into_chunks(text)
    
    # Utiliser tous les segments comme contexte
    context = " ".join(chunks)
    
    # Générer une réponse
    response = generate_response_gemini(context, question)
    return response

# Interface Gradio
def interface_rag(pdf_file, question):
    return process_question(pdf_file, question)

# Création de l'interface Gradio
interface = gr.Interface(
    fn=interface_rag,
    inputs=[
        gr.File(label="Télécharger un PDF"),
        gr.Textbox(label="Posez votre question")
    ],
    outputs="text",
    title="Système RAG pour les PDF avec Google Gemini 2",
    description="Posez une question basée sur un PDF et obtenez une réponse contextuelle."
)

# Lancer l'interface
if __name__ == "__main__":
    interface.launch()
