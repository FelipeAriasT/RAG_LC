
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML, Markdown
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import os

# Añadir el directorio raíz al path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Intentar importar el módulo de utilidades
try:
    import utils_rag as ur
except ImportError:
    print("⚠️ No se pudo importar utils_rag. Las imágenes de páginas no funcionarán.")
    # Crear un módulo ficticio para evitar errores
    class UtilsRAGMock:
        def plot_multiple_pages(self, *args, **kwargs):
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Error: No se pudo cargar la imagen", 
                   horizontalalignment='center', verticalalignment='center')
            return fig
    ur = UtilsRAGMock()

class ChatManager:
    """Clase para gestionar el historial y visualización del chat - Optimizada para VS Code"""
    
    def __init__(self):
        self.messages = []
        
        # Optimizado para VS Code: usar un output más grande y con scroll
        self.output_widget = widgets.Output(
            layout=widgets.Layout(
                border='1px solid #ddd',
                padding='10px',
                overflow_y='auto',
                min_height='400px',
                max_height='600px',
                width='99%'
            )
        )
        
        # Mejorar el contenedor para VS Code
        self.container = widgets.VBox([
            widgets.HTML("<h2>Chat</h2>"),
            self.output_widget
        ], layout=widgets.Layout(width='99%', margin='10px 0px'))
        
        self.current_sources = None
        self.source_viewers = {}
    
    def display_chat_ui(self):
        """Muestra el widget de chat"""
        display(self.container)
        return self
    
    def add_user_message(self, content):
        """Añade un mensaje de usuario al historial"""
        self.messages.append({
            "role": "user",
            "content": content,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        self._render_messages()
        return self
    
    def add_assistant_message(self, content, sources=None):
        """Añade un mensaje del asistente al historial"""
        self.messages.append({
            "role": "assistant",
            "content": content,
            "sources": sources,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        self.current_sources = sources
        self._render_messages()
        return self
    
    def _render_messages(self):
        """Renderiza todos los mensajes - Optimizado para VS Code"""
        with self.output_widget:
            clear_output(wait=True)
            
            if not self.messages:
                display(HTML("<p><i>Escribe una consulta para comenzar...</i></p>"))
                return
                
            for msg_idx, msg in enumerate(self.messages):
                if msg["role"] == "user":
                    # Mensaje de usuario más simple para VS Code
                    display(Markdown(f"""
                    **👤 Tú - {msg['timestamp']}**
                    
                    {msg['content']}
                    
                    ---
                    """))
                else:
                    # Mensaje del asistente con markdown para VS Code
                    display(Markdown(f"""
                    **🤖 Asistente - {msg['timestamp']}**
                    
                    {msg['content']}
                    """))
                    
                    # Si hay fuentes, mostrar botón para ellas
                    if msg.get("sources") and len(msg["sources"]) > 0:
                        sources_btn_id = f"sources_btn_{msg_idx}"
                        
                        # Botón más simple
                        sources_toggle = widgets.ToggleButton(
                            description=f"📚 Ver {len(msg['sources'])} fuentes",
                            value=False,
                            button_style='info',
                            icon='eye',
                            layout=widgets.Layout(width='200px')
                        )
                        
                        # Output para fuentes
                        sources_output = widgets.Output(
                            layout=widgets.Layout(
                                border='1px solid #ddd',
                                padding='10px',
                                margin='5px 0px',
                                max_height='400px',
                                overflow_y='auto'
                            )
                        )
                        
                        # Crear el widget para mostrar las fuentes
                        self._render_sources_in_output(sources_output, msg["sources"])
                        
                        # Función para mostrar/ocultar fuentes
                        def on_toggle_sources(change):
                            if change['new']:
                                sources_output.layout.display = 'block'
                                sources_toggle.description = f"📚 Ocultar fuentes"
                            else:
                                sources_output.layout.display = 'none'
                                sources_toggle.description = f"📚 Ver {len(msg['sources'])} fuentes"
                        
                        # Registrar la función
                        sources_toggle.observe(on_toggle_sources, names='value')
                        
                        # Ocultar las fuentes inicialmente
                        sources_output.layout.display = 'none'
                        
                        # Mostrar los widgets
                        display(sources_toggle)
                        display(sources_output)
                    
                    # Separador entre mensajes
                    display(HTML("<hr>"))
    
    def _render_sources_in_output(self, output_widget, sources):
        """Renderiza las fuentes - Optimizado para VS Code"""
        with output_widget:
            for idx, citation in enumerate(sources):
                context_key = list(citation.keys())[0]
                citation_data = citation[context_key]
                
                # Usar markdown para mejor visualización en VS Code
                display(Markdown(f"""
                ### {context_key}
                
                **Archivo:** {citation_data['file_name']}
                
                **Páginas:** {', '.join(map(str, citation_data['page_numbers']))}
                """))
                
                # Botón para mostrar las imágenes
                show_btn = widgets.Button(
                    description="Mostrar páginas",
                    button_style='primary',
                    layout=widgets.Layout(width='150px')
                )
                
                # Output para las imágenes
                images_output = widgets.Output(
                    layout=widgets.Layout(
                        margin='10px 0px',
                        max_height='500px',
                        overflow_y='auto'
                    )
                )
                
                # Función para mostrar las imágenes
                def on_show_pages(b, citation_data=citation_data, output=images_output):
                    with output:
                        clear_output(wait=True)
                        try:
                            data_path = os.path.join(root_dir, "data/docs/")
                            fig = ur.plot_multiple_pages(
                                data_path + citation_data['file_name'], 
                                citation_data['page_numbers']
                            )
                            plt.tight_layout()
                            plt.show()
                        except Exception as e:
                            print(f"Error al cargar las imágenes: {str(e)}")
                            plt.figure(figsize=(8, 4))
                            plt.text(0.5, 0.5, f"No se pudieron cargar las imágenes: {str(e)}", 
                                    ha='center', va='center', wrap=True)
                            plt.axis('off')
                            plt.show()
                
                # Asignar la función al botón
                show_btn.on_click(lambda b, data=citation_data, out=images_output: on_show_pages(b, data, out))
                
                # Mostrar el botón y el output para las imágenes
                display(show_btn)
                display(images_output)
                
                # Separador
                display(HTML("<hr>"))
    
    def get_messages(self):
        """Retorna todos los mensajes del historial"""
        return self.messages
    
    def clear_history(self):
        """Limpia el historial de mensajes"""
        self.messages = []
        self.source_viewers = {}
        self.current_sources = None
        self._render_messages()
        return self
    
    def get_current_sources(self):
        """Retorna las fuentes actuales"""
        return self.current_sources
    
    def hide(self):
        """Oculta el widget de chat"""
        self.container.layout.visibility = 'hidden'
        self.container.layout.display = 'none'
        return self
    
    def show(self):
        """Muestra el widget de chat"""
        self.container.layout.visibility = 'visible'
        self.container.layout.display = 'flex'
        return self