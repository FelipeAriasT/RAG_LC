import streamlit as st
import time
import os
import sys
from typing import Dict, Any, List

# Agregar directorio raíz al path para poder importar utilidades
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Importar módulos personalizados
from app.chat_manager import ChatManager
from app.ui_components import (
    render_chat_messages, render_sidebar_controls, 
    render_sources_panel, render_chat_input,
    render_sources_toggle_button
)
from app.progress_callback import ProgressCallback

# Importar el módulo de RAG
sys.path.append(os.path.join(root_dir))
from rag import run_rag

def initialize_session_state():
    """Inicializa el estado de la sesión si no existe"""
    if 'chat_manager' not in st.session_state:
        st.session_state.chat_manager = ChatManager()
    if 'show_sources' not in st.session_state:
        st.session_state.show_sources = False
    if 'current_sources' not in st.session_state:
        st.session_state.current_sources = None
    if 'is_processing' not in st.session_state:
        st.session_state.is_processing = False
    if 'last_model' not in st.session_state:
        st.session_state.last_model = None
    if 'last_n_results' not in st.session_state:
        st.session_state.last_n_results = None
    if 'progress_callback' not in st.session_state:
        st.session_state.progress_callback = ProgressCallback()
    if 'query_count' not in st.session_state:
        st.session_state.query_count = 0

def process_rag_query(query: str, params: Dict[str, Any], progress_callback: ProgressCallback) -> Dict[str, Any]:
    """Procesa una consulta RAG con los parámetros dados"""
    try:
        result = run_rag(
            query=query,
            use_query_expansion=params["use_query_expansion"],
            use_reranker=params["use_reranker"],
            model=params["model"],
            n_results=params["n_results"],
            progress_callback=progress_callback
        )
        
        # Extraer la respuesta y las citas
        answer = result.get("answer", "Lo siento, no puedo generar una respuesta en este momento.")
        citations = result.get("citations", [])
        
        return {
            "status": "success",
            "answer": answer,
            "citations": citations,
            "cost": result.get("cost", 0)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error al procesar la consulta: {str(e)}"
        }

def toggle_sources():
    """Cambia el estado de visualización de fuentes"""
    st.session_state.show_sources = not st.session_state.show_sources

def check_model_change(current_model, current_n_results):
    """Comprueba si el modelo ha cambiado y reinicia el chat si es necesario"""
    model_changed = False
    
    # Verificar si es la primera vez o si alguno de los parámetros ha cambiado
    if (st.session_state.last_model is None or 
        st.session_state.last_model != current_model or
        st.session_state.last_n_results != current_n_results):
        
        # Actualizar los valores del último modelo y n_results
        st.session_state.last_model = current_model
        st.session_state.last_n_results = current_n_results
        model_changed = True
    
    return model_changed

def main():
    """Función principal de la aplicación Streamlit"""
    st.set_page_config(
        page_title="RAG Sistema",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Estilo CSS para botones flotantes y personalización
    st.markdown("""
    <style>
    .sources-panel {
        position: fixed;
        right: 0;
        top: 0;
        bottom: 0;
        width: 400px;
        background-color: white;
        padding: 20px;
        z-index: 1000;
        box-shadow: -5px 0px 10px rgba(0,0,0,0.1);
        overflow-y: auto;
    }
    .float-button {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 1001;
    }
    .stButton button {
        border-radius: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Inicializar estado de sesión
    initialize_session_state()
    
    # Configuración en la barra lateral
    params = render_sidebar_controls()
    
    # Verificar si el modelo ha cambiado
    if check_model_change(params["model"], params["n_results"]):
        # Si el modelo cambió, reiniciar el chat
        st.session_state.chat_manager = ChatManager()
        st.session_state.show_sources = False
        st.session_state.current_sources = None
        st.session_state.is_processing = False
        st.session_state.progress_callback = ProgressCallback()
        st.session_state.query_count = 0
        st.info(f"Se ha seleccionado un nuevo modelo: {params['model']} o número de resultados: {params['n_results']}. El chat ha sido reiniciado.")
    
    # Botón para reiniciar el chat
    if st.sidebar.button("Reiniciar Chat"):
        st.session_state.chat_manager = ChatManager()
        st.session_state.show_sources = False
        st.session_state.current_sources = None
        st.session_state.is_processing = False
        st.session_state.progress_callback = ProgressCallback()
        st.session_state.query_count = 0
        st.rerun()
    
    # Mostrar información del modelo actual
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Modelo actual:** {params['model']}")
    st.sidebar.markdown(f"**Número de resultados:** {params['n_results']}")
    
    # Contenedor principal para el chat
    st.title("Sistema de RAG")
    
    # Botón para mostrar/ocultar panel de fuentes (fuera de cualquier columna)
    if st.session_state.current_sources:
        sources_button_clicked = render_sources_toggle_button(
            st.session_state.show_sources, 
            st.session_state.query_count
        )
        
        if sources_button_clicked:
            toggle_sources()
            st.rerun()
    
    # Definir contenedores principales
    chat_container = st.container()
    
    # Configuración de columnas para chat y panel lateral
    with chat_container:
        # Mostrar mensajes del chat
        render_chat_messages(st.session_state.chat_manager.get_messages())
        
        # Si está procesando, mostrar progreso en tiempo real
        if st.session_state.is_processing:
            progress_placeholder = st.empty()
            st.session_state.progress_callback.set_placeholder(progress_placeholder)
    
    # Panel lateral de fuentes (sólo si está visible y hay fuentes)
    if st.session_state.show_sources and st.session_state.current_sources:
        with st.sidebar:
            st.markdown("## Panel de Fuentes")
            
            # Botón en el panel para cerrar las fuentes
            if st.button("❌ Cerrar panel", key="close_sources_panel"):
                st.session_state.show_sources = False
                st.rerun()
            
            st.markdown("---")
            
            # Renderizar el panel de fuentes en la barra lateral
            render_sources_panel(st.session_state.current_sources, st.sidebar)
    
    # Entrada de chat en la parte inferior
    query, submit_button = render_chat_input()
    
    # Procesar la consulta cuando se envía
    if submit_button and query and not st.session_state.is_processing:
        # Incrementar contador de consultas para generar IDs únicos
        st.session_state.query_count += 1
        
        # Agregar mensaje del usuario
        st.session_state.chat_manager.add_user_message(query)
        
        # Marcar como procesando y reiniciar callback de progreso
        st.session_state.is_processing = True
        st.session_state.progress_callback = ProgressCallback()
        st.rerun()
    
    # Si está procesando, realizar la consulta RAG
    if st.session_state.is_processing and not st.session_state.chat_manager.current_response:
        # Obtener la última consulta del usuario
        user_messages = [m for m in st.session_state.chat_manager.get_messages() if m["role"] == "user"]
        if user_messages:
            last_query = user_messages[-1]["content"]
            
            # Crear un placeholder para el progreso
            progress_placeholder = st.empty()
            st.session_state.progress_callback.set_placeholder(progress_placeholder)
            
            # Procesar la consulta
            result = process_rag_query(last_query, params, st.session_state.progress_callback)
            
            if result["status"] == "success":
                # Guardar la respuesta y las fuentes
                st.session_state.chat_manager.add_assistant_message(
                    result["answer"],
                    result["citations"]
                )
                st.session_state.current_sources = result["citations"]
                
                # Mostrar información de costo
                st.sidebar.markdown(f"**Costo estimado:** ${result['cost']:.5f}")
            else:
                # Mostrar error
                st.session_state.chat_manager.add_assistant_message(
                    f"Error: {result.get('message', 'Ocurrió un error desconocido.')}"
                )
            
            # Marcar como no procesando
            st.session_state.is_processing = False
            st.rerun()

if __name__ == "__main__":
    main()