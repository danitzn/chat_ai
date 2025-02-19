import ollama

def ask_deepseek(question, document_path=None):
    """
    Envía una pregunta a DeepSeek R1 usando Ollama y devuelve la respuesta.
    Si se proporciona un documento, inclúyelo en la pregunta.
    """
    if document_path:
        with open(document_path, "r") as file:
            document_content = file.read()
        question = f"Documento: {document_content}\n\nPregunta: {question}"
    
    
    response = ollama.generate(
        model="deepseek-r1:1.5b",  
        prompt=question
    )
    
    return response["response"]