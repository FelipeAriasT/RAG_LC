
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
import sys
import os
import threading
import time

# Añadir el directorio raíz al path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Importar el módulo RAG con manejo de errores
try:
    from rag import run_rag
except ImportError:
    print("⚠️ No se pudo importar el módulo RAG. Algunas funcionalidades no estarán disponibles.")
    # Función mock para desarrollo
    def run_rag(query, use_query_expansion=False, use_reranker=True, 
                model="gemini-1.5-flash", n_results=5, progress_callback=None):
        """Función mock para pruebas cuando no está disponible el RAG real"""
        if progress_callback:
            for state in ['initializing', 'query_expansion', 'document_retrieval', 
                        'response_generation', 'citation_validation', 
                        'citation_extraction', 'final_validation', 'complete']:
                progress_callback.update_progress(state)
                time.sleep(0.5)
                
        return {
            "answer": f"Respuesta simulada para la consulta: '{query}'.\n\nEsta es una respuesta de prueba.",
            "citations": [
                {"Contexto 1": {"file_name": "ejemplo.pdf", "page_numbers": [1, 2]}}
            ],
            "cost": 0.00001
        }

class RAGJupyterApp:
    """Aplicación principal - Optimizada para VS Code"""
    
    def __init__(self, chat_manager, progress_callback, ui_components):
        self.chat_manager = chat_manager
        self.progress_callback = progress_callback
        self.ui = ui_components
        self.is_processing = False
        self.last_model = None
        self.last_n_results = None
        self.total_cost = 0.0
        
        # Configurar callbacks
        self.ui.set_submit_callback(self._on_submit)
        self.ui.set_reset_callback(self._on_reset)
        
        # Crear layout principal - mejorado para VS Code
        self.main_container = widgets.VBox([
            widgets.HTML("<h1 style='text-align:center;'>Sistema RAG Interactivo</h1>"),
            widgets.HTML("<hr style='margin-bottom:20px;'>"),
            
            # Usar tabs para mejor organización en VS Code
            widgets.Tab([
                # Tab 1: Chat y consulta
                widgets.VBox([
                    widgets.HBox([
                        # Área de chat (70%)
                        self.chat_manager.container,
                    ], layout=widgets.Layout(width='100%')),
                    
                    # Área de progreso
                    self.progress_callback.container,
                    
                    # Área de consulta
                    widgets.HTML("<h3>Nueva Consulta</h3>"),
                    self.ui.query_container
                ]),
                
                # Tab 2: Configuración
                self.ui.controls_container
            ])
        ])
        
        # Establecer los títulos de las pestañas
        self.main_container.children[2].set_title(0, "Chat")
        self.main_container.children[2].set_title(1, "Configuración")
    
    def display(self):
        """Muestra la aplicación completa"""
        display(HTML("""
        <style>
        .jupyter-widgets-output-area .output_scroll {
            height: auto !important;
            max-height: 400px !important;
        }
        .jp-OutputArea-output {
            overflow-y: auto !important;
            max-height: none !important;
        }
        .jp-RenderedHTMLCommon img {
            max-width: 100% !important;
        }
        </style>
        """))
        display(self.main_container)
        self.progress_callback.hide()
        return self
    
    def _on_submit(self, b):
        """Maneja el evento de envío de consulta"""
        if self.is_processing:
            return
        
        query = self.ui.get_query()
        if not query:
            # Mostrar un mensaje de error en lugar de simplemente retornar
            with self.chat_manager.output_widget:
                display(HTML("<div style='color:red;'>Por favor ingresa una consulta.</div>"))
            return
        
        # Verificar si cambió el modelo o n_results
        config = self.ui.get_config()
        if (self.last_model is not None and 
            (self.last_model != config["model"] or 
             self.last_n_results != config["n_results"])):
            # Reiniciar chat si cambió la configuración
            self.chat_manager.clear_history()
            self.progress_callback.reset()
            self.total_cost = 0.0
            self.ui.update_cost(self.total_cost)
            
            # Informar al usuario del cambio
            with self.chat_manager.output_widget:
                display(HTML(f"<div style='color:blue;'>Configuración cambiada: Nuevo modelo {config['model']} o resultados {config['n_results']}. Chat reiniciado.</div>"))
        
        # Actualizar última configuración
        self.last_model = config["model"]
        self.last_n_results = config["n_results"]
        
        # Desactivar controles durante el procesamiento
        self.ui.disable_controls()
        self.is_processing = True
        
        # Añadir mensaje del usuario
        self.chat_manager.add_user_message(query)
        self.ui.clear_query()
        
        # Mostrar progreso
        self.progress_callback.show().reset()
        
        # Procesar la consulta en un hilo separado - mejorado para VS Code
        def safe_processing_thread():
            try:
                self._process_query(query, config)
            except Exception as e:
                # Capturar cualquier excepción no manejada
                error_message = f"Error inesperado: {str(e)}"
                self.chat_manager.add_assistant_message(error_message)
                self.ui.enable_controls()
                self.is_processing = False
                self.progress_callback.hide()
        
        thread = threading.Thread(target=safe_processing_thread)
        thread.daemon = True
        thread.start()
    
    def _process_query(self, query, config):
        """Procesa la consulta en un hilo separado"""
        try:
            # Ejecutar RAG
            result = run_rag(
                query=query,
                use_query_expansion=config["use_query_expansion"],
                use_reranker=config["use_reranker"],
                model=config["model"],
                n_results=config["n_results"],
                progress_callback=self.progress_callback
            )
            
            # Añadir respuesta del asistente
            answer = result.get("answer", "Lo siento, no puedo generar una respuesta en este momento.")
            citations = result.get("citations", [])
            cost = result.get("cost", 0)
            
            self.chat_manager.add_assistant_message(answer, citations)
            
            # Actualizar costo
            self.total_cost += cost
            self.ui.update_cost(self.total_cost)
            
        except Exception as e:
            # Manejar errores con más detalles
            import traceback
            error_details = traceback.format_exc()
            error_message = f"Error al procesar la consulta: {str(e)}\n\n```\n{error_details}\n```"
            self.chat_manager.add_assistant_message(error_message)
        
        finally:
            # Habilitar controles y ocultar progreso
            self.ui.enable_controls()
            self.is_processing = False
            self.progress_callback.hide()
    
    def _on_reset(self, b):
        """Maneja el evento de reinicio de chat"""
        if self.is_processing:
            return
        
        self.chat_manager.clear_history()
        self.progress_callback.reset().hide()
        self.total_cost = 0.0
        self.ui.update_cost(self.total_cost)
        self.ui.clear_query()
        
        # Mostrar mensaje de confirmación
        with self.chat_manager.output_widget:
            display(HTML("<div style='color:green;'>Chat reiniciado correctamente.</div>"))