from langchain_text_splitters.markdown import MarkdownTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama.llms import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.documents import Document
from markitdown import MarkItDown
from typing import List
from pydantic import BaseModel, ValidationError, validator
from ollama import chat
import os
from pyvis.network import Network
from pypdf import PdfReader
from markdown import markdown
from xhtml2pdf import pisa
import json
import re
import time
from typing import List, Optional
import random


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


def extract_references(text):
    pattern = re.compile(r"References\s*(.*)", re.IGNORECASE | re.DOTALL)
    match = pattern.search(text)

    if match:
        return match.group(1).strip()
    else:
        return ""
    

class Reference(BaseModel):
    title: str
    authors: List[str]
    date: str
    journal: Optional[str] = None

class ReferenceGraph(BaseModel):
    references: List[Reference]


def extract_references_with_prompts(article_text, llm_name):
    references = extract_references(article_text)
    print(references)
    if not references:
        return []

    prompt = f"""
    You are an expert in extracting references from scientific articles.
    Here are the references found in the article:
    {references}

    Please extract all references in the following JSON format:
    [
        {{
            "title": "Title of the reference",
            "authors": ["Author 1", "Author 2"],
            "date": "Publication Date",
            "journal": "Journal name (if available)",
        }},
        ...
    ]

    Only include the references section from the article and follow the format exactly.
    """

    response = chat(
        messages=[{"role": "user", "content": prompt}],
        model=llm_name,
        format=ReferenceGraph.model_json_schema()
    )

    print("\n model output", response.message.content)

    return response.message.content


def generate_random_color(excluded_colors=None):
    """Generate a random hex color code, excluding specified colors."""
    if excluded_colors is None:
        excluded_colors = []
    while True:
        color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
        if color not in excluded_colors:
            return color


def save_references_graph(article_title, references, file_name_only: str):
    if isinstance(references, str):
        try:
            references = json.loads(references).get("references", [])
        except json.JSONDecodeError as e:
            print(f"Error parsing references JSON: {e}")
            return

    net = Network(height="800px", width="100%", notebook=False, directed=False)

    net.add_node(
        article_title,
        label=article_title,
        shape="dot",
        color="#779ECB",
        title=f"Article: {article_title}"
    )

    all_ref_names = set()

    for ref in references:
        if not isinstance(ref, dict):
            print(f"Invalid reference type: {type(ref)} - {ref}")
            continue

        # Safely extract details from the reference
        title = ref.get("title", "Unknown Title")
        date = ref.get("date", "Unknown Date")
        authors = ref.get("authors", [])
        journal = ref.get("journal", "Unknown Journal")

        tooltip = (
            f"Title: {title}\n"
            f"Date: {date}\n"
            f"Authors: {', '.join(authors) or 'Unknown Authors'}\n"
            f"Journal: {journal}"
        )

        node_color = generate_random_color(["#779ECB"])
        net.add_node(
            title,
            label=title,
            shape="dot",
            color=node_color,
            title=tooltip
        )
        net.add_edge(article_title, title, color="#000")
        all_ref_names.add(title)

    out_dir = "graphs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{file_name_only}_graph.html")
    net.save_graph(out_path)

    print(f"Graph saved to: {out_path}")
    return out_path