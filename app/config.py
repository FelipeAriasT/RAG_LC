# Configuración para la aplicación Streamlit
import os

# Modelos disponibles para la aplicación
AVAILABLE_MODELS = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro"
]

# Configuración predeterminada
DEFAULT_MODEL = "gemini-2.0-flash"
DEFAULT_N_RESULTS = 5
DEFAULT_USE_QUERY_EXPANSION = False
DEFAULT_USE_RERANKER = True

# Rutas de datos
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/docs/")

# Mensajes de estado para el streaming del flujo RAG real
RAG_PROGRESS_STATES = {
    "initializing": "Inicializando el procesamiento de la consulta",
    "query_expansion": "Expandiendo la consulta para mejorar los resultados",
    "document_retrieval": "Recuperando documentos relevantes de la base de conocimiento",
    "response_generation": "Generando una respuesta basada en los documentos recuperados",
    "citation_validation": "Validando citas y referencias en la respuesta",
    "citation_extraction": "Extrayendo información de citas para presentación",
    "final_validation": "Realizando validación final de la respuesta",
    "complete": "¡Respuesta lista!"
}