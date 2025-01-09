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


class Concept(BaseModel):
    name: str
    description: str
    related_concepts: List[str]

class ConceptGraph(BaseModel):
    article_title: str
    concepts: List[Concept]


def extract_concepts_from_article(article_text: str) -> ConceptGraph:
    prompt_content = f"""
    You are an Expert AI that identifies all concepts from a scientific article.
    Article text: ""{article_text}""

    Return JSON that follows this schema exactly:
    {ConceptGraph.model_json_schema()}

    - article_title: (string) The title or short name of this article
    - concepts: (array) a list of:
    - name: (string) concept name
    - description: (string) short explanation
    - related_concepts: (array of strings) all concept names related to this concept
    """

    response = chat(
        messages=[
            {
                "role": "user",
                "content": prompt_content,
            }
        ],
        model="llama3.2:3b",
        format=ConceptGraph.model_json_schema(),
    )

    json_text = response.message.content

    concept_graph = ConceptGraph.model_validate_json(json_text)

    return concept_graph


def show_concept_graph_in_notebook(concept_graph: ConceptGraph, out_html="concept_graph.html"):
    net = Network(height="600px", width="100%", notebook=False, directed=False)

    net.add_node(
        concept_graph.article_title,
        label=concept_graph.article_title,
        shape='dot',
        color='#FF6961',
        title=f"Article: {concept_graph.article_title}",
    )

    # Add each concept node + edges to the article
    for c in concept_graph.concepts:
        net.add_node(
            c.name,
            label=c.name,
            shape='dot',
            color='#77DD77',
            title=c.description,
        )
        net.add_edge(concept_graph.article_title, c.name)

    # Add edges for related concepts
    all_concept_names = {cc.name for cc in concept_graph.concepts}
    for c in concept_graph.concepts:
        for related_name in c.related_concepts:
            if related_name in all_concept_names and related_name != c.name:
                net.add_edge(c.name, related_name)

    out_dir = "graphs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_html)
    net.save_graph(out_path)


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