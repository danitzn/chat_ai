from deepseek_local import ask_deepseek

# Prueba sin documento
response = ask_deepseek("Hola, ¿cómo estás?")
print("Respuesta sin documento:", response)

# Prueba con documento
response = ask_deepseek("¿Qué dice el documento?", "uploads/documento.txt")
print("Respuesta con documento:", response)