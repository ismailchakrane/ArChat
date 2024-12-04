from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Charger le PDF
loader = PyPDFLoader("data/FA208976_20241120_163611.pdf")
documents = loader.load()

# Supprimer les doublons au niveau du contenu
unique_documents = {doc.page_content: doc for doc in documents}
documents = list(unique_documents.values())

# Diviser le contenu en morceaux gérables (chunks)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)  # Ajuster la taille
docs = text_splitter.split_documents(documents)

# Vérifier la longueur des chunks
for i, doc in enumerate(docs[:5]):
    print(f"Chunk {i+1} length: {len(doc.page_content.split())} tokens")

# Utiliser HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Créer le vectorstore en évitant les doublons
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="data/vectorstore"
)

# Persister les données
vectorstore.persist()

print("Vectorstore créé et persisté avec succès.")
