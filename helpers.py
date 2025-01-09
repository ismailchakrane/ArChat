from langchain_text_splitters.markdown import MarkdownTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama.llms import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.documents import Document
from markitdown import MarkItDown
from typing import List
from pydantic import BaseModel
from ollama import chat
import os
from pyvis.network import Network
from pypdf import PdfReader
from markdown import markdown
from xhtml2pdf import pisa

import warnings
warnings.filterwarnings("ignore")

def get_available_pdfs(directory="data"):
    results = process_pdf_directory(directory)

    title_to_file = {}
    for item in results:
        title = item["title"]
        filename = item["filename"]
        title_to_file[title] = filename

    return title_to_file


def get_txt_content(pdf_path):
    md = MarkItDown()
    result = md.convert(pdf_path)
    return result.text_content


def retrieve_from_vectorstore(vectorstore, query, k=20):
    docs = vectorstore.similarity_search(query, k=k)
    return "\n".join([doc.page_content for doc in docs])


def load_pdf_to_vectorstore(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    unique_docs = {}
    for doc in documents:
        unique_docs[doc.page_content] = doc
    documents = list(unique_docs.values())

    all_text = "\n".join(doc.page_content for doc in documents)

    text_splitter = MarkdownTextSplitter(chunk_size=2048, chunk_overlap=150)
    chunk_strings = text_splitter.split_text(all_text)

    all_chunks_as_docs = [Document(page_content=txt) for txt in chunk_strings]

    embeddings = HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
    vectorstore = FAISS.from_documents(all_chunks_as_docs, embeddings)

    return vectorstore


def save_uploaded_file(uploaded_file, directory="data"):
    with open(os.path.join(directory, uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    return uploaded_file.name


class Reference(BaseModel):
    title: str
    date: str
    authors: List[str]
    related_references: List[str] = []

class ReferenceGraph(BaseModel):
    article_title: str
    references: List[Reference]


def extract_references_from_article(article_text: str) -> ReferenceGraph:
    prompt_content = f"""
    You are an Expert AI that identifies all references from a scientific article.

    ARTICLE:
    {article_text}

    Return JSON that follows this schema exactly:
    {ReferenceGraph.model_json_schema()}

    Explanation of the schema:
    - article_title: (string) The title or short name of this article
    - references: (array) a list of references. Each reference has:
      - title: (string) The reference's own title
      - date: (string) The publication date
      - authors: (array of strings) The authors' names
      - related_references: (array of strings) references that are related (by citation or mention)
    """

    response = chat(
        messages=[{"role": "user", "content": prompt_content}],
        model="llama3.2:3b",
        format=ReferenceGraph.model_json_schema(),
    )
    json_text = response.message.content

    reference_graph = ReferenceGraph.model_validate_json(json_text)
    return reference_graph


def reference_graph_to_html(
    reference_graph: ReferenceGraph,
    out_html="references_graph.html"
):
    net = Network(height="600px", width="100%", notebook=False, directed=False)

    # Center node for the article itself
    net.add_node(
        reference_graph.article_title,
        label=reference_graph.article_title,
        shape="dot",
        color="#FF6961",  # e.g., a red-ish color for the article center
        title=f"Article: {reference_graph.article_title}"
    )

    # A small color palette or random approach for references
    # (Here we define a few colors and cycle through them)
    color_palette = ["#77DD77", "#779ECB", "#F49AC2", "#CFCFC4", "#FFB347", "#AEC6CF"]

    all_ref_names = set()
    for idx, ref in enumerate(reference_graph.references):
        # Each reference node
        node_color = color_palette[idx % len(color_palette)]
        # Show date & authors in the 'title' for hover
        tooltip = (
            f"Title: {ref.title}\n"
            f"Date: {ref.date}\n"
            f"Authors: {', '.join(ref.authors)}"
        )
        net.add_node(
            ref.title,
            label=ref.title,  # node label = reference title
            shape="dot",
            color=node_color,
            title=tooltip
        )
        # Edge from the article to this reference
        net.add_edge(
            reference_graph.article_title,
            ref.title,
            color="#555555"  # e.g. a dark gray for edges
        )
        all_ref_names.add(ref.title)

    # If references have 'related_references', connect them
    for ref in reference_graph.references:
        for rel_ref in ref.related_references:
            if rel_ref in all_ref_names and rel_ref != ref.title:
                net.add_edge(ref.title, rel_ref, color="#555555")

    # Save to 'graphs/' subfolder
    out_dir = "graphs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_html)
    net.save_graph(out_path)

    print(f"Graph HTML saved to: {out_path}")
    return out_path

def extract_pdf_title(file_path):
    try:
        reader = PdfReader(file_path)
        metadata = reader.metadata
        
        if metadata and '/Title' in metadata and metadata['/Title'].strip():
            return metadata['/Title']
        
        if len(reader.pages) > 0:
            first_page = reader.pages[0]
            text = first_page.extract_text().strip()
            if text:
                lines = text.split('\n')[:2]
                combined_title = " ".join(line.strip() for line in lines if line.strip())
                return combined_title if combined_title else "Inconnu"
        
        return file_path
    except Exception as e:
        print(f"Erreur avec le fichier {file_path}: {e}")
        return "Erreur lors de l'extraction"


def process_pdf_directory(directory):
    data = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.pdf'):
                file_path = os.path.join(root, file)
                title = extract_pdf_title(file_path)
                data.append({
                    "filename": file,
                    "title": title
                })
    return data


def markdown_to_pdf(markdown_string, output_file="output.pdf"):
    html_content = markdown(markdown_string)

    out_dir = "summary"
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, output_file)

    with open(out_path, "wb") as pdf_file:
        pisa_status = pisa.CreatePDF(html_content, dest=pdf_file)

    if pisa_status.err:
        print("Error: Unable to create PDF")
    else:
        print(f"PDF successfully created: {out_path}")

import json
from ollama import chat

def extract_references_with_prompts(article_text):
    """
    Utilise un modèle de langage pour extraire les références d'un texte d'article scientifique.

    Args:
        article_text (str): Le texte complet de l'article.

    Returns:
        list: Une liste de dictionnaires représentant les références extraites.
    """
    # Préparer le prompt pour extraire les références
    prompt = f"""
    You are an expert in extracting references from scientific articles.

    ARTICLE TEXT:
    {article_text}

    Please extract all references in the following JSON format:
    [
        {{
            "title": "Title of the reference",
            "authors": ["Author 1", "Author 2"],
            "date": "Publication Date"
        }},
        ...
    ]

    Only include the references section from the article and follow the format exactly.
    """

    # Utilisation du modèle de langage pour répondre au prompt
    response = chat(
        messages=[{"role": "user", "content": prompt}],
        model="llama3.2:3b"  # Remplacez par le modèle que vous utilisez
    )
    # Parser la réponse JSON
    try:
        references = json.loads(response.message.content)
    except json.JSONDecodeError:
        references = []  # Retourner une liste vide en cas d'erreur de parsing

    return references
def save_references_to_json(references, output_file="references.json"):
    """
    Sauvegarde les références extraites au format JSON.

    Args:
        references (list): Liste des références extraites.
        output_file (str): Nom du fichier de sortie JSON.

    Returns:
        str: Le chemin du fichier JSON sauvegardé.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(references, f, indent=2, ensure_ascii=False)
    return output_file



