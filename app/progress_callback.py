from typing import Dict, Any, Callable, Optional
import streamlit as st
from app.config import RAG_PROGRESS_STATES

class ProgressCallback:
    """
    Clase para gestionar callbacks de progreso del flujo RAG.
    Permite reportar el progreso actual a Streamlit.
    """
    
    def __init__(self):
        self.current_state = "initializing"
        self.placeholder = None
        self.progress_callback = None
    
    def set_placeholder(self, placeholder):
        """Establece un placeholder de Streamlit para actualizar"""
        self.placeholder = placeholder
        return self
    
    def set_callback(self, callback_fn: Callable[[str, Dict[str, Any]], None]):
        """
        Establece una función de callback personalizada
        La función debe aceptar (state_name, state_details)
        """
        self.progress_callback = callback_fn
        return self
        
    def update_progress(self, state_name: str, details: Optional[Dict[str, Any]] = None):
        """
        Actualiza el progreso actual
        
        Args:
            state_name: Nombre del estado actual (debe estar en RAG_PROGRESS_STATES)
            details: Detalles adicionales sobre el estado
        """
        if state_name not in RAG_PROGRESS_STATES:
            state_name = "initializing"
            
        self.current_state = state_name
        
        # Si tenemos un callback definido, llámalo
        if self.progress_callback:
            self.progress_callback(state_name, details or {})
            
        # Si tenemos un placeholder de Streamlit, actualízalo
        if self.placeholder:
            with self.placeholder.container():
                self._render_progress(state_name, details or {})
    
    def _render_progress(self, state_name: str, details: Dict[str, Any]):
        """Renderiza el progreso actual en Streamlit"""
        message = RAG_PROGRESS_STATES.get(state_name, "Procesando...")
        
        col1, col2 = st.columns([1, 11])
        
        with col1:
            st.spinner("")  # Esto muestra el icono de carga giratorio
        with col2:
            st.markdown(f"**{message}**")
            
            # Si hay detalles adicionales, mostrarlos
            if details and details.get("additional_info"):
                st.markdown(f"*{details['additional_info']}*")
                
        # Mostrar el estado actual como texto 
        states = list(RAG_PROGRESS_STATES.keys())
        current_index = states.index(state_name) if state_name in states else 0
        
        # Mostrar los pasos completados como texto
        for i, state in enumerate(states[:current_index]):
            st.markdown(f"✅ {RAG_PROGRESS_STATES[state]}")
        
        # Mostrar el paso actual con el spinner
        if state_name != "complete":
            st.markdown(f"⏳ {RAG_PROGRESS_STATES[state_name]}")

    def get_current_state(self):
        """Obtiene el estado actual del progreso"""
        return self.current_state