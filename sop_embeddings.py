import fitz 
import os
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path="./pdf_chroma_db") 
collection = chroma_client.get_or_create_collection(name="pdf_embeddings")

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

def store_pdf_embeddings_in_vector_db(pdf_text, pdf_id):
    text_chunks = pdf_text.split("\n")
    
    embeddings = embedding_model.encode(text_chunks).tolist()
    
    for idx, (chunk, embedding) in enumerate(zip(text_chunks, embeddings)):
        collection.add(
            ids=[f"{pdf_id}-{idx}"], 
            embeddings=[embedding], 
            documents=[chunk]  
        )
    
    print(f"Stored {len(text_chunks)} chunks of PDF text in ChromaDB.")


pdf_path = r"document/sop.pdf" 
pdf_text = extract_text_from_pdf(pdf_path)
store_pdf_embeddings_in_vector_db(pdf_text, pdf_id="sop_001")
