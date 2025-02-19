import faiss
from fastapi import FastAPI, File, UploadFile, HTTPException
import numpy as np
from pydantic import BaseModel
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import os

from deepseek_local import ask_deepseek  # Importar BaseModel desde Pydantic

# Definir el modelo de datos para el cuerpo de la solicitud
class ChatRequest(BaseModel):
    message: str

app = FastAPI()

# Variables globales para almacenar el documento procesado
document_index = None
document_sentences = []

# Cargar un modelo de embeddings (por ejemplo, 'all-MiniLM-L6-v2')
# model = SentenceTransformer('all-MiniLM-L6-v2') cambio por un modelo mas pequeño
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Función para extraer texto de un PDF
# def extract_text_from_pdf(file_path: str) -> str:
#     reader = PdfReader(file_path)
#     text = ""
#     for page in reader.pages:
#         text += page.extract_text()
#     return text


def extract_text_from_pdf(file_path: str, max_pages: int = 10) -> str:
    reader = PdfReader(file_path)
    text = ""
    for i, page in enumerate(reader.pages):
        if i >= max_pages:
            break
        text += page.extract_text()
    return text

# Función para crear un índice FAISS
def create_faiss_index(text: str):
    # Dividir el texto en oraciones o fragmentos
    sentences = text.split(". ")  # Simple división por oraciones
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Generar embeddings para cada oración
    embeddings = model.encode(sentences)
    
    # Crear un índice FAISS
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # Índice de búsqueda por similitud
    index.add(np.array(embeddings))
    
    return index, sentences

# Función para buscar información relevante en el índice FAISS
def search_in_index(index, sentences, query: str, top_k: int = 3):
    # Generar embedding para la consulta
    query_embedding = model.encode([query])
    
    # Buscar los fragmentos más relevantes
    distances, indices = index.search(np.array(query_embedding), top_k)
    
    # Devolver los fragmentos más relevantes
    results = [sentences[i] for i in indices[0]]
    return results

# Ruta para subir documentos
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    global document_index, document_sentences
    
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Solo se permiten archivos PDF")
    
    file_location = f"uploads/{file.filename}"
    os.makedirs("uploads", exist_ok=True)  # Crea la carpeta si no existe
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())
    
    # Extraer texto del PDF
    text = extract_text_from_pdf(file_location)
    
    # Crear un índice FAISS con el texto extraído
    document_index, document_sentences = create_faiss_index(text)
    
    return {"filename": file.filename, "location": file_location}

# Ruta para interactuar con el chat
@app.post("/chat/")
async def chat(request: ChatRequest):
    global document_index, document_sentences
    
    if document_index is None or not document_sentences:
        raise HTTPException(status_code=400, detail="No se ha subido ningún documento")
    
    try:
        # 1. Buscar información relevante en el documento
        query = request.message
        relevant_texts = search_in_index(document_index, document_sentences, query)
        
        # 2. Combinar la información relevante con la consulta
        context = " ".join(relevant_texts)
        augmented_query = f"Contexto: {context}\n\nPregunta: {query}"
        
        # 3. Generar una respuesta usando DeepSeek R1
        response = ask_deepseek(augmented_query)
        
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))