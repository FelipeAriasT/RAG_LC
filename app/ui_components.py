import streamlit as st
import time
import base64
from PIL import Image
import io
from typing import List, Dict, Optional
import os
import sys

# Agregar directorio raíz al path para poder importar utilidades
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import utils_rag as ur

def render_chat_messages(messages: List[Dict]):
    """Renderiza todos los mensajes del chat"""
    for msg_idx, msg in enumerate(messages):
        if msg["role"] == "user":
            render_user_message(msg["content"], msg["timestamp"], msg_idx)
        else:
            render_assistant_message(msg["content"], msg["timestamp"], msg.get("sources"), msg_idx)

def render_user_message(content: str, timestamp: str, msg_idx: int):
    """Renderiza un mensaje del usuario"""
    col1, col2 = st.columns([1, 12])
    
    with col1:
        st.markdown("👤")
    with col2:
        st.markdown(f"**Tú** - {timestamp}")
        st.markdown(content)
    
    st.markdown("---")

def render_assistant_message(content: str, timestamp: str, sources: Optional[List[Dict]] = None, msg_idx: int = 0):
    """Renderiza un mensaje del asistente"""
    col1, col2 = st.columns([1, 12])
    
    with col1:
        st.markdown("🤖")
    with col2:
        st.markdown(f"**Asistente** - {timestamp}")
        st.markdown(content)
        
        # Si hay fuentes, mostramos un pequeño botón de citas junto al mensaje
        if sources and len(sources) > 0:
            # Crear un identificador único para este mensaje
            msg_id = f"msg_{msg_idx}_{timestamp.replace(':', '')}"
            
            # Estilo CSS personalizado para el botón de fuentes
            st.markdown("""
            <style>
            .source-btn {
                display: inline-block;
                padding: 2px 8px;
                background-color: #f0f2f6;
                color: #262730;
                border-radius: 4px;
                border: 1px solid #ddd;
                font-size: 0.8em;
                margin-top: 5px;
                cursor: pointer;
                text-decoration: none;
            }
            .source-btn:hover {
                background-color: #e6e9ef;
                text-decoration: none;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Mini botón para mostrar fuentes
            st.markdown(
                f'<div><a class="source-btn" href="#" id="source-btn-{msg_id}" '
                f'onclick="document.getElementById(\'sources-panel-btn\').click(); return false;">'
                f'📚 Ver {len(sources)} fuentes</a></div>',
                unsafe_allow_html=True
            )
            
            # También mantenemos el expandible para visualización directa
            with st.expander("Detalles de fuentes"):
                for src_idx, citation in enumerate(sources):
                    contexto_key = list(citation.keys())[0]
                    citation_data = citation[contexto_key]
                    
                    st.markdown(f"**{contexto_key}**")
                    st.markdown(f"Archivo: {citation_data['file_name']}")
                    st.markdown(f"Páginas: {', '.join(map(str, citation_data['page_numbers']))}")
                    
                    # Botón para mostrar las imágenes de las páginas
                    # Crear una clave única basada en el índice del mensaje, el índice de la fuente y el contenido
                    unique_key = f"show_pages_msg{msg_idx}_src{src_idx}_{contexto_key}"
                    
                    if st.button(f"Mostrar páginas - {contexto_key}", key=unique_key):
                        data_path = os.path.join(root_dir, "data/docs/")
                        try:
                            fig = ur.plot_multiple_pages(
                                data_path + citation_data['file_name'], 
                                citation_data['page_numbers']
                            )
                            st.pyplot(fig)
                            
                        except Exception as e:
                            st.error(f"Error al cargar las imágenes: {str(e)}")
    
    st.markdown("---")

def render_sidebar_controls():
    """Renderiza los controles en la barra lateral"""
    from app.config import AVAILABLE_MODELS, DEFAULT_MODEL, DEFAULT_N_RESULTS
    from app.config import DEFAULT_USE_QUERY_EXPANSION, DEFAULT_USE_RERANKER
    
    st.sidebar.title("Configuración de RAG")
    
    model = st.sidebar.selectbox(
        "Modelo", 
        options=AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index(DEFAULT_MODEL) if DEFAULT_MODEL in AVAILABLE_MODELS else 0
    )
    
    n_results = st.sidebar.slider(
        "Número de resultados", 
        min_value=1, 
        max_value=10, 
        value=DEFAULT_N_RESULTS
    )
    
    use_query_expansion = st.sidebar.checkbox(
        "Usar expansión de consulta", 
        value=DEFAULT_USE_QUERY_EXPANSION
    )
    
    use_reranker = st.sidebar.checkbox(
        "Usar reordenador", 
        value=DEFAULT_USE_RERANKER
    )
    
    return {
        "model": model,
        "n_results": n_results,
        "use_query_expansion": use_query_expansion,
        "use_reranker": use_reranker
    }

def render_sources_panel(sources: List[Dict], container=None):
    """Renderiza el panel de fuentes en un contenedor específico"""
    target = container if container else st
    
    target.title("Fuentes Citadas")
    
    for idx, citation in enumerate(sources):
        contexto_key = list(citation.keys())[0]
        citation_data = citation[contexto_key]
        
        target.markdown(f"### {contexto_key}")
        target.markdown(f"**Archivo:** {citation_data['file_name']}")
        target.markdown(f"**Páginas:** {', '.join(map(str, citation_data['page_numbers']))}")
        
        # Crear una clave única para cada botón
        unique_key = f"sidebar_src{idx}_{contexto_key}"
        
        # Mostrar las imágenes
        data_path = os.path.join(root_dir, "data/docs/")
        try:
            fig = ur.plot_multiple_pages(
                data_path + citation_data['file_name'], 
                citation_data['page_numbers']
            )
            target.pyplot(fig)
        except Exception as e:
            target.error(f"Error al cargar las imágenes: {str(e)}")
        
        target.markdown("---")

def render_chat_input():
    """Renderiza el área de entrada del chat"""
    with st.container():
        col1, col2 = st.columns([5, 1])
        
        with col1:
            query = st.text_input("Escribe tu consulta aquí", key="query_input")
        
        with col2:
            submit_button = st.button("Enviar", use_container_width=True)
        
        return query, submit_button

def render_sources_toggle_button(show_sources, query_count):
    """Renderiza un botón para mostrar/ocultar el panel de fuentes"""
    # Usar un ID único basado en el número de consulta para evitar conflictos
    button_id = f"sources-panel-btn-{query_count}"
    
    # Botón flotante estilizado para mostrar/ocultar fuentes
    col1, col2, col3 = st.columns([2, 1, 7])
    with col2:
        result = st.button(
            "📚" if not show_sources else "❌", 
            key=button_id,
            help="Mostrar/Ocultar panel de fuentes"
        )
    
    return result