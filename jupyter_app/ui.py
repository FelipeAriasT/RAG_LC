import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
import sys
import os

# Añadir el directorio raíz al path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

try:
    from app.config import AVAILABLE_MODELS, DEFAULT_MODEL, DEFAULT_N_RESULTS
    from app.config import DEFAULT_USE_QUERY_EXPANSION, DEFAULT_USE_RERANKER
except ImportError:
    # Si no existe el módulo, definimos valores por defecto
    AVAILABLE_MODELS = [
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-pro"
    ]
    DEFAULT_MODEL = "gemini-2.0-flash"
    DEFAULT_N_RESULTS = 5
    DEFAULT_USE_QUERY_EXPANSION = False
    DEFAULT_USE_RERANKER = True

class UIComponents:
    """Clase para gestionar componentes de interfaz de usuario"""
    
    def __init__(self):
        # Componentes para la consulta
        self.query_input = widgets.Text(
            description='Consulta:',
            placeholder='Escribe tu consulta aquí...',
            layout=widgets.Layout(width='80%')
        )
        
        self.submit_button = widgets.Button(
            description='Enviar',
            button_style='primary',
            icon='paper-plane',
            layout=widgets.Layout(width='150px')
        )
        
        self.query_container = widgets.HBox([
            self.query_input, 
            self.submit_button
        ])
        
        # Componentes para la configuración
        self.model_selector = widgets.Dropdown(
            options=AVAILABLE_MODELS,
            value=DEFAULT_MODEL,
            description='Modelo:',
            disabled=False,
            layout=widgets.Layout(width='80%')
        )
        
        self.n_results_slider = widgets.IntSlider(
            value=DEFAULT_N_RESULTS,
            min=1,
            max=10,
            step=1,
            description='Resultados:',
            disabled=False,
            layout=widgets.Layout(width='80%')
        )
        
        self.query_expansion_checkbox = widgets.Checkbox(
            value=DEFAULT_USE_QUERY_EXPANSION,
            description='Expansión de consulta',
            disabled=False
        )
        
        self.reranker_checkbox = widgets.Checkbox(
            value=DEFAULT_USE_RERANKER,
            description='Reordenador',
            disabled=False
        )
        
        # Agrupar componentes de configuración
        self.config_container = widgets.VBox([
            widgets.HTML("<h3>Configuración RAG</h3>"),
            self.model_selector,
            self.n_results_slider,
            self.query_expansion_checkbox,
            self.reranker_checkbox
        ])
        
        # Botón para reiniciar chat
        self.reset_button = widgets.Button(
            description='Reiniciar Chat',
            button_style='warning',
            icon='refresh',
            layout=widgets.Layout(width='150px')
        )
        
        # Contenedor para la información de costo
        self.cost_info = widgets.HTML(value="<p><b>Costo estimado:</b> $0.00000</p>")
        
        # Agrupar componentes de información
        self.info_container = widgets.VBox([
            self.reset_button,
            widgets.HTML("<hr>"),
            self.cost_info
        ])
        
        # Contenedor principal para los controles
        self.controls_container = widgets.VBox([
            self.config_container,
            widgets.HTML("<hr>"),
            self.info_container
        ])
    
    def display_query_ui(self):
        """Muestra la interfaz de consulta"""
        display(self.query_container)
        return self
    
    def display_controls_ui(self):
        """Muestra la interfaz de controles"""
        display(self.controls_container)
        return self
    
    def set_submit_callback(self, callback):
        """Establece el callback para el botón de enviar"""
        self.submit_button.on_click(callback)
        return self
    
    def set_reset_callback(self, callback):
        """Establece el callback para el botón de reinicio"""
        self.reset_button.on_click(callback)
        return self
    
    def get_query(self):
        """Obtiene la consulta actual"""
        return self.query_input.value
    
    def clear_query(self):
        """Limpia la consulta actual"""
        self.query_input.value = ''
        return self
    
    def get_config(self):
        """Obtiene la configuración actual"""
        return {
            "model": self.model_selector.value,
            "n_results": self.n_results_slider.value,
            "use_query_expansion": self.query_expansion_checkbox.value,
            "use_reranker": self.reranker_checkbox.value
        }
    
    def update_cost(self, cost):
        """Actualiza la información de costo"""
        self.cost_info.value = f"<p><b>Costo estimado:</b> ${cost:.5f}</p>"
        return self
    
    def disable_controls(self, disable=True):
        """Habilita o deshabilita los controles"""
        self.model_selector.disabled = disable
        self.n_results_slider.disabled = disable
        self.query_expansion_checkbox.disabled = disable
        self.reranker_checkbox.disabled = disable
        self.query_input.disabled = disable
        self.submit_button.disabled = disable
        return self
    
    def enable_controls(self):
        """Habilita los controles"""
        return self.disable_controls(False)