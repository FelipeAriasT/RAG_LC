
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
import sys
import os

# Añadir el directorio raíz al path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

try:
    from app.config import RAG_PROGRESS_STATES
except ImportError:
    # Si no existe el módulo, definimos unos estados por defecto
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

class JupyterProgressCallback:
    """
    Clase para gestionar callbacks de progreso del flujo RAG en Jupyter.
    Optimizada para Visual Studio Code.
    """
    
    def __init__(self):
        self.current_state = "initializing"
        
        # Usar un widget Output más simple para VS Code
        self.output_widget = widgets.Output(
            layout=widgets.Layout(
                border='1px solid #ddd',
                padding='10px',
                overflow_y='auto',
                max_height='150px'
            )
        )
        
        # Barra de progreso simplificada
        self.progress_widget = widgets.IntProgress(
            value=0,
            min=0,
            max=len(RAG_PROGRESS_STATES),
            description='Progreso:',
            bar_style='info',
            orientation='horizontal',
            layout=widgets.Layout(width='99%', height='20px')
        )
        
        # Etiqueta de estado como texto simple para mejor compatibilidad
        self.state_label = widgets.Label(
            value="Estado: Inicializando...",
            layout=widgets.Layout(margin='10px 0px')
        )
        
        self.states_completed = []
        
        # Organizar contenedor principal
        self.container = widgets.VBox([
            self.state_label,
            self.progress_widget,
            self.output_widget
        ], layout=widgets.Layout(margin='10px 0px'))
    
    def display_progress_ui(self):
        """Muestra los widgets de progreso"""
        display(self.container)
        return self
    
    def update_progress(self, state_name, details=None):
        """Actualiza el progreso actual - Optimizado para VS Code"""
        if state_name not in RAG_PROGRESS_STATES:
            state_name = "initializing"
            
        self.current_state = state_name
        if state_name not in self.states_completed:
            self.states_completed.append(state_name)
        
        # Calcular el índice del estado actual
        states = list(RAG_PROGRESS_STATES.keys())
        current_index = states.index(state_name) if state_name in states else 0
        
        # Actualizar el widget de progreso - método seguro para VS Code
        self.progress_widget.value = current_index + 1
        
        # Actualizar la etiqueta de estado
        message = RAG_PROGRESS_STATES.get(state_name, "Procesando...")
        self.state_label.value = f"Estado: {message}"
        
        # Mostrar el progreso en el widget de salida
        with self.output_widget:
            clear_output(wait=True)
            for i, state in enumerate(states[:current_index]):
                if state in self.states_completed:
                    print(f"✅ {RAG_PROGRESS_STATES[state]}")
            print(f"⏳ {RAG_PROGRESS_STATES[state_name]}")
            
            # Si hay detalles adicionales, mostrarlos
            if details and 'additional_info' in details:
                print(f"Info: {details['additional_info']}")
    
    def get_current_state(self):
        """Obtiene el estado actual del progreso"""
        return self.current_state
    
    def reset(self):
        """Reinicia el estado del progreso"""
        self.current_state = "initializing"
        self.progress_widget.value = 0
        self.state_label.value = "Estado: Listo para procesar"
        self.states_completed = []
        with self.output_widget:
            clear_output(wait=True)
        return self
    
    def hide(self):
        """Oculta los widgets de progreso"""
        self.container.layout.visibility = 'hidden'
        self.container.layout.display = 'none'
        return self
    
    def show(self):
        """Muestra los widgets de progreso"""
        self.container.layout.visibility = 'visible'
        self.container.layout.display = 'flex'
        return self