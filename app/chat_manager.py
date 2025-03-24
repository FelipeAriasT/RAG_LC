from datetime import datetime
from typing import List, Dict, Any

class ChatManager:
    """Clase para gestionar el historial y estado del chat"""
    
    def __init__(self):
        self.reset_chat()
    
    def reset_chat(self):
        """Reinicia el historial del chat"""
        self.messages = []
        self.current_response = None
        self.current_sources = None
        
    def add_user_message(self, message: str):
        """Añade un mensaje del usuario al historial"""
        self.messages.append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
    
    def add_assistant_message(self, message: str, sources: List[Dict] = None):
        """Añade un mensaje del asistente al historial"""
        self.messages.append({
            "role": "assistant",
            "content": message,
            "sources": sources,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        self.current_response = None
        self.current_sources = None
    
    def set_current_response(self, message: str):
        """Establece la respuesta actual en proceso"""
        self.current_response = message
    
    def set_current_sources(self, sources: List[Dict]):
        """Establece las fuentes de la respuesta actual"""
        self.current_sources = sources
    
    def get_messages(self):
        """Retorna todos los mensajes del historial"""
        return self.messages
    
    def get_current_state(self):
        """Retorna el estado actual de la respuesta en proceso"""
        return {
            "response": self.current_response,
            "sources": self.current_sources
        }